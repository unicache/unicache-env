from gym.envs.registration import register

for dataset in ['iqiyi', 'movielens']:
    for capDivCont in ['0.00001', '0.0001', '0.001', '0.01']: # Using string because there is precision problems of %f 
        for sampleSizeStr in [100, 1000, 10000, 'full']: # Size of full set = 233045
            for version in range(16):
                sampleSize = sampleSizeStr if sampleSizeStr != 'full' else None
                register(
                    id = 'cache-%s-%s-%s-v%d'%(dataset, capDivCont, sampleSizeStr, version), # '-v\d' is required by gym 
                    kwargs = {'dataset': dataset, 'capDivCont': float(capDivCont), 'sampleSize': sampleSize, 'version': version},
                    entry_point = 'unicache_env.envs:Env'
                )

