// Save the committed scene locally, even if its server cache entry has expired.
export function recipeForScenario(scenario) {
  const recipe=structuredClone(scenario.resolved_request);
  recipe.provenance=Object.fromEntries(['core_version','rng_policy','numpy_version'].map(key=>[key,scenario.provenance[key]]));
  if(scenario.provenance.event_mixing_policy)recipe.provenance.event_mixing_policy=scenario.provenance.event_mixing_policy;
  return recipe;
}
export function saveJSON(value,filename) {
  const blob=new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'});
  const url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=filename;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
