"""Massive canonical acceptance and independent conditional count sampling.

This is a count-only integration of the shared rest-frame p4 law, not a new
momentum sampler. A Lorentz boost changes d³p by E/E*, hence the lab density in
(pT,y,phi) is pT E* exp[-(E*-m)/T]. The phi integral uses scaled I0/I1 functions.
"""
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from scipy.special import i0e, i1e, kve

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "toys/boltzmann"))
import clumps
from generate_thermal_h5 import CONFIGS, sample_glauber_and_multiplicity

MASS = clumps.PROTON_MASS
RNG_POLICY = "bes-counts-seedsequence-pcg64-v1"


def rng(seed, role, *keys):
    root = int(clumps.seed(str(seed)))
    entropy = [root & 0xffffffff, root >> 32, 0x42455343, 1, role, *map(int, keys)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


@lru_cache(maxsize=16)
def nodes(order):
    return np.polynomial.legendre.leggauss(order)


def acceptance(temperature, beta, cos_theta, y_max, pt_min=.4, pt_max=2., order=64):
    """Full-azimuth probability of |y|<y_max and pt_min<pT<pt_max.

    No eta preselection is imposed. Symmetric rapidity acceptance makes this
    even in cos(theta). Supports vector directions in bounded working chunks.
    """
    if not .001 <= temperature <= .5 or not 0 <= beta <= .95:
        raise ValueError("Invalid T or beta")
    if not 0 < y_max <= 3 or not 0 <= pt_min < pt_max:
        raise ValueError("Invalid analysis acceptance")
    direction = np.atleast_1d(np.asarray(cos_theta, dtype=float))
    if not np.isfinite(direction).all() or np.any(abs(direction) > 1):
        raise ValueError("Invalid direction cosine")
    x, weight = nodes(order)
    pt = (pt_max-pt_min)*.5*x+(pt_max+pt_min)*.5
    wp = weight*(pt_max-pt_min)*.5
    y = y_max*x; wy = y_max*weight
    mt = np.sqrt(MASS*MASS+pt*pt)
    energy = mt[:, None]*np.cosh(y)
    pz = mt[:, None]*np.sinh(y)
    gamma = 1/np.sqrt(1-beta*beta)
    norm = 4*np.pi*MASS*MASS*temperature*kve(2, MASS/temperature)
    out = []
    for start in range(0, len(direction), 32):
        u = direction[start:start+32]
        bz = beta*u[:, None, None]
        bt = beta*np.sqrt(1-u*u)[:, None, None]
        q = energy[None, :, :]-bz*pz[None, :, :]
        a = gamma*bt*pt[None, :, None]/temperature
        integrand = (2*np.pi*gamma*pt[None, :, None]
                     * (q*i0e(a)-bt*pt[None, :, None]*i1e(a))
                     * np.exp((MASS-gamma*q)/temperature+a))
        out.extend(np.einsum('ijk,j,k->i', integrand, wp, wy)/norm)
    result = np.asarray(out)
    if np.any(result < -1e-12) or np.any(result > 1+1e-10):
        raise ArithmeticError("Acceptance quadrature outside [0,1]")
    return result


def acceptance_table(temperature, beta, windows, pt_min=.4, pt_max=2., tolerance=2e-6):
    """Converged quadrature and direction interpolation, checked at midpoints."""
    grid = np.linspace(0, 1, 65)
    errors = []
    values = np.array([acceptance(temperature,beta,grid,w,pt_min,pt_max,64) for w in windows])
    fine = np.array([acceptance(temperature,beta,grid,w,pt_min,pt_max,96) for w in windows])
    quad_error = float(np.max(abs(values-fine)))
    if quad_error > tolerance/5:
        finest = np.array([acceptance(temperature,beta,grid,w,pt_min,pt_max,144) for w in windows])
        quad_error = float(np.max(abs(fine-finest))); fine = finest
        if quad_error > tolerance/5:
            raise ArithmeticError("Quadrature did not converge")
    values = fine
    order = 144 if quad_error > tolerance/50 else 96
    while True:
        mid = (grid[:-1]+grid[1:])/2
        actual = np.array([acceptance(temperature,beta,mid,w,pt_min,pt_max,order) for w in windows])
        error = float(np.max(abs(actual-(values[:, :-1]+values[:, 1:])/2)))
        errors.append(error)
        if error <= tolerance:
            break
        if len(grid) >= 2049:
            raise ArithmeticError("Direction interpolation did not converge")
        combined = np.empty((len(windows), len(grid)+len(mid)))
        combined[:, ::2] = values; combined[:, 1::2] = actual
        grid = np.linspace(0, 1, combined.shape[1]); values = combined
    if np.any(np.diff(values, axis=0) < -tolerance):
        raise ArithmeticError("Rapidity probabilities are not nested")
    return {"grid": grid, "values": values, "quadrature_error": quad_error,
            "interpolation_error": errors[-1], "tolerance": tolerance}


def selected_geometry(n_events, seed, centrality=(0., .05), chunk_size=131072):
    """Independent minimum-bias draws, selecting before any particle generation.

    Fixed chunk streams preserve event prefixes when increasing sample size.
    Only geometry/multiplicity from the legacy BES model are reused.
    """
    low, high = centrality
    if not 0 <= low < high <= 1:
        raise ValueError("Invalid centrality")
    fields = [[] for _ in range(6)]; have = 0; chunk = 0
    while have < n_events:
        c,b,np_,nc,m = sample_glauber_and_multiplicity(CONFIGS['bes'],chunk_size,rng(seed,1,chunk))
        keep = np.flatnonzero((c >= low)&(c < high))[:n_events-have]
        for dest,array in zip(fields,[c,b,np_,nc,m,chunk*chunk_size+np.arange(chunk_size)]):
            dest.append(array[keep])
        have += len(keep); chunk += 1
    return dict(zip(['centrality','b','npart','ncoll','emitted','event_id'],[np.concatenate(x) for x in fields]))


def background_counts(emitted, probabilities, seed):
    """Exact nested multinomial proton/antiproton counts for fixed T/beta=0."""
    species = np.array(CONFIGS['bes'].species_probs); species /= species.sum()
    shells = np.diff(np.r_[0., probabilities])
    p = np.r_[species[4]*shells,species[5]*shells,1-(species[4]+species[5])*probabilities[-1]]
    draws = rng(seed,2).multinomial(emitted, p)
    n = len(probabilities)
    return np.cumsum(draws[:,:n],axis=1), np.cumsum(draws[:,n:2*n],axis=1)


def source_counts(n_events, n_values, clumps_per_event, table, direction, seed):
    """Nested conditional Bernoulli counts with shared or independent directions.

    This common-uniform coupling preserves each control's distribution. With
    fixed T/beta and transverse boosts, shared/mixed counts agree identically:
    their full-azimuth acceptance probability is independent of the boost phi.
    """
    maximum = max(n_values); windows = len(table['values'])
    counts = {m: np.zeros((n_events, windows, len(n_values)), dtype=np.uint16)
              for m in ['shared_boost','event_mixed']}
    grid, values = table['grid'], table['values']
    for clump in range(clumps_per_event):
        uniform = np.column_stack([rng(seed,10,clump,d).random(n_events) for d in range(maximum)])
        common = rng(seed,11,clump).random(n_events) if direction=='isotropic_3d' else np.zeros(n_events)
        donors = np.column_stack([rng(seed,12,clump,d).random(n_events) for d in range(maximum)]) if direction=='isotropic_3d' else np.zeros((n_events,maximum))
        for w in range(windows):
            a = np.interp(common,grid,values[w])[:,None]
            b = np.interp(donors,grid,values[w])
            for model,probability in [('shared_boost',a),('event_mixed',b)]:
                cumulative = np.cumsum(uniform < probability,axis=1)
                counts[model][:,w,:] += cumulative[:,np.array(n_values)-1].astype(np.uint16)
    return counts


def population_cumulants(raw):
    a,b,c,d = np.asarray(raw)
    return np.array([a,b-a*a,c-3*b*a+2*a**3,d-4*c*a-3*b*b+12*b*a*a-6*a**4])


def cumulants_to_raw(c):
    a,b,c_,d = c
    return np.array([a,b+a*a,c_+3*b*a+a**3,d+4*c_*a+3*b*b+6*b*a*a+a**4])


def source_prediction(n, k, f, table, direction, model, window):
    a = table['values'][window]
    if direction=='transverse':
        ar = [a[0]**r for r in range(1,5)]
    else:
        ar = [float(np.trapezoid(a**r,table['grid'])) for r in range(1,5)]
    if model=='poisson':
        return np.full(4,f*k*n*ar[0])
    if model=='event_mixed':
        ar = [ar[0]**r for r in range(1,5)]
    falling = [np.prod([n-j for j in range(r)]) if n>=r else 0 for r in range(1,5)]
    z = np.array(falling)*ar
    raw = [z[0],z[1]+z[0],z[2]+3*z[1]+z[0],z[3]+6*z[2]+7*z[1]+z[0]]
    summed = cumulants_to_raw(k*population_cumulants(raw))
    return population_cumulants(f*summed)
