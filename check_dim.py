import json
for f in ['concept_space_75k.json', 'concept_space_40k.json', 'concept_space_60k.json']:
    with open('real_data/' + f) as fp:
        d = json.load(fp)
    print(f + ': dim=' + str(d.get('dim')) + ' latent_dim=' + str(d.get('fractal', {}).get('latent_dim', '?')))
