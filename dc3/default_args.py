def method_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 2
    defaults['simpleIneq'] = 1
    defaults['simpleEq'] = 1
    defaults['simpleEx'] = 10000
    #defaults['nonconvexVar'] = 100
    #defaults['nonconvexIneq'] = 50
    #defaults['nonconvexEq'] = 50
    #defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50


    if prob_type == 'nonlinear':
        defaults['epochs'] = 100
        defaults['batchSize'] = 200
        defaults['lr'] = 0.1
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 100          # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 5
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-4
        defaults['corrMomentum'] = 0.5   
        defaults['qtySamples'] = 2         
    elif 'dc_wss' in prob_type:
        defaults['epochs'] = 10
        defaults['batchSize'] = 50
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 64
        defaults['softWeight'] = 50 # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = False
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'full'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 5
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 0.3          # use 1e-5 if useCompl=False
        defaults['corrMomentum'] = 0.5 
    elif 'nonlinear_2ineq' in prob_type:
        defaults['epochs'] = 100
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10          # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 5
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-4
        defaults['corrMomentum'] = 0.5   
        defaults['qtySamples'] = 2

       
    else:
        raise NotImplementedError

    return defaults