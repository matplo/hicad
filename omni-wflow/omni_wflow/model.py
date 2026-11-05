"""OmniLearned PET2 with explicit masks and periodic whole-event neighborhoods.

All trainable encoder/classifier layers belong to installed OmniLearned. The
local orchestration corrects jet-coordinate/padding assumptions without editing it.
"""
import torch
from torch import nn
from omnilearned.network import PET2


def device(name='auto'):
    if name!='auto':return torch.device(name)
    if torch.cuda.is_available():return torch.device('cuda')
    if torch.backends.mps.is_available():
        try:
            (torch.ones(2,device='mps')@torch.ones(2,device='mps')).cpu()
            return torch.device('mps')
        except RuntimeError:pass
    return torch.device('cpu')


def collate(items):
    xs,ys=zip(*items);size=max(1,max(map(len,xs)));width=xs[0].shape[-1]
    x=torch.zeros(len(xs),size,width);mask=torch.zeros(len(xs),size,1,dtype=torch.bool)
    for i,a in enumerate(xs):x[i,:len(a)]=torch.as_tensor(a);mask[i,:len(a)]=True
    return x,mask,torch.as_tensor(ys,dtype=torch.long)


def periodic_local(block,x,mask):
    b,n,f=x.shape;k=min(block.K,max(1,n-1))
    deta=x[:,:,None,0]-x[:,None,:,0];dphi=x[:,:,None,1]-x[:,None,:,1]
    distance=deta.square()+torch.atan2(dphi.sin(),dphi.cos()).square()
    distance=distance.masked_fill(~mask[:,None,:,0],1e9)
    distance=distance.masked_fill(torch.eye(n,device=x.device,dtype=torch.bool)[None],2e9)
    indices=distance.topk(k,dim=-1,largest=False).indices
    batch=torch.arange(b,device=x.device)[:,None,None]
    neighbors=x[batch,indices];valid=mask[batch,indices]&mask[:,:,None,:]
    valid=valid & (indices!=torch.arange(n,device=x.device)[None,:,None])[...,None]
    local=x[:,:,None,:]-neighbors
    local=torch.cat([local[...,:1],torch.atan2(local[...,1:2].sin(),local[...,1:2].cos()),local[...,2:]],dim=-1)
    local=local.reshape(b*n,k,f);valid=valid.reshape(b*n,k,1)
    bias=(~(valid&valid.transpose(1,2))).repeat_interleave(block.num_heads,0).float()*-1e9
    z=block.mlp(local)*valid
    for layer in block.in_blocks:z=layer(z,mask=valid,attn_mask=bias)
    z=z.reshape(b,n,k,-1);valid=valid.reshape(b,n,k,1)
    return (z*valid).sum(2)/valid.sum(2).clamp_min(1)*mask


class EventPET(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=spec
        self.pet=PET2(input_dim=spec['features'],use_int=False,local_int=False,
            num_coord=2,mode='classifier',num_classes=2,base_dim=spec['dim'],
            num_transformers=spec['depth'],num_transformers_head=spec['head_depth'],
            num_heads=spec['heads'],num_tokens=4,K=spec['neighbors'],mlp_drop=0.,attn_drop=0.)
        self.denoise=nn.Linear(spec['dim'],2)
        self.angle_embed=nn.Linear(2,spec['dim'],bias=False) if spec.get('periodic_embedding',False) else None

    def encode(self,x,mask):
        body=self.pet.body;b=len(x)
        embedding_input=x
        if self.angle_embed is not None:
            embedding_input=torch.cat([x[:,:,:1],torch.zeros_like(x[:,:,1:2]),x[:,:,2:]],dim=-1)
        embedded,_=body.embed(embedding_input,mask)
        if self.angle_embed is not None:
            embedded=embedded+self.angle_embed(torch.stack([x[:,:,1].sin(),x[:,:,1].cos()],-1))*mask
        z=embedded+periodic_local(body.local_physics,x,mask)
        z=torch.cat([body.token.expand(b,-1,-1),z],dim=1)
        valid=torch.cat([torch.ones(b,body.num_tokens,1,dtype=torch.bool,device=x.device),mask],dim=1)
        bias=(~(valid&valid.transpose(1,2))).repeat_interleave(body.num_heads,0).float()*-1e9
        for layer in body.in_blocks:z=layer(z,mask=valid,attn_mask=bias)
        return body.norm(z)*valid,valid

    def forward(self,x,mask):
        z,valid=self.encode(x,mask);head=self.pet.classifier
        head_mask=valid[:,head.num_tokens:].clone()
        head_mask[:,0,0] |= ~head_mask.any(dim=(1,2))
        for layer in head.in_blocks:z=layer(z,mask=head_mask)
        return head.out(head.fc(z[:,:head.num_tokens].reshape(len(x),-1)))

    def reconstruct(self,x,mask):
        z,_=self.encode(x,mask)
        return self.denoise(z[:,self.pet.body.num_tokens:])

    def initialize_encoder(self,path):
        archive=torch.load(path,map_location='cpu',weights_only=True)
        source=archive.get('state',archive.get('model',archive));target=self.state_dict()
        source={k.removeprefix('module.'):v for k,v in source.items()}
        report={'loaded':[],'excluded':[],'angular_adapter':'restored' if 'angle_embed.weight' in source else 'fresh'}
        if self.angle_embed is not None and 'angle_embed.weight' in source:
            self.angle_embed.load_state_dict({'weight':source['angle_embed.weight']})
        for key,value in source.items():
            mapped=key if key.startswith('pet.') else 'pet.'+key
            if not mapped.startswith('pet.body.'):
                report['excluded'].append(key);continue
            if mapped not in target:
                if any(s in mapped for s in ['time_embed','MPFourier']):report['excluded'].append(key);continue
                raise ValueError('Unexpected encoder weight: '+key)
            if target[mapped].shape!=value.shape:raise ValueError('Encoder architecture/feature mismatch: '+key)
            target[mapped]=value;report['loaded'].append(mapped)
        missing=set(k for k in target if k.startswith('pet.body.'))-set(report['loaded'])
        if missing:raise ValueError('Missing encoder weights: '+str(sorted(missing)))
        self.load_state_dict(target,strict=True)
        return report
