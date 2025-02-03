def method_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 3
    defaults['simpleIneq'] = 2
    defaults['simpleEq'] = 0
    defaults['simpleEx'] = 10
    #defaults['nonconvexVar'] = 100
    #defaults['nonconvexIneq'] = 50
    #defaults['nonconvexEq'] = 50
    #defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50


    if prob_type == 'nonlinear':
        defaults['epochs'] = 10
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 100          # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = False
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'full'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 10
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5                    
    elif 'dc_wss' in prob_type:
        defaults['epochs'] = 10
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10             # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = False
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'full'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 5
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-4           # use 1e-5 if useCompl=False
        defaults['corrMomentum'] = 0.5 
    else:
        raise NotImplementedError

    return defaults