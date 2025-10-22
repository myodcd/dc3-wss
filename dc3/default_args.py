def method_default_args(prob_type):
    defaults = {}
    defaults["simpleVar"] = 2
    defaults["simpleIneq"] = 1
    defaults["simpleEq"] = 1
    defaults["simpleEx"] = 10000
    # defaults['nonconvexVar'] = 100
    # defaults['nonconvexIneq'] = 50
    # defaults['nonconvexEq'] = 50
    # defaults['nonconvexEx'] = 10000
    defaults["saveAllStats"] = True
    defaults["resultsSaveFreq"] = 50

    defaults["computer_type"] = "personal"

    if prob_type == "nonlinear":
        defaults["epochs"] = 50
        defaults["batchSize"] = 200
        defaults["lr"] = 0.1
        defaults["hiddenSize"] = 200
        defaults["softWeight"] = 100  # use 100 if useCompl=False
        defaults["softWeightEqFrac"] = 0.5
        defaults["useCompl"] = True
        defaults["useTrainCorr"] = True
        defaults["useTestCorr"] = True
        defaults["corrMode"] = "partial"  # use 'full' if useCompl=False
        defaults["corrTrainSteps"] = 5
        defaults["corrTestMaxSteps"] = 5
        defaults["corrEps"] = 1e-4
        defaults["corrLr"] = 1e-4  # use 1e-5 if useCompl=False
        defaults["corrLrStart"] = 0
        defaults["corrLrDuration"] = 0
        defaults["corrMomentum"] = 0.5
        defaults["qtySamples"] = 2

    elif "dc_wss" in prob_type:
        defaults["epochs"] = 15
        defaults["batchSize"] = 200  # 16
        defaults["lr"] = 1e-3  # 1e-4
        defaults["hiddenSize"] = 200
        defaults["softWeight"] = 100  # use 100 if useCompl=False
        defaults["softWeightEqFrac"] = 0.5  # RODADA 1

        defaults["softWeightEqFracStart"] = 0.2  # 1.7 # 0.2
        defaults["softWeightEqFracDuration"] = 0.8  # 1.1 # 0.8

        defaults["useCompl"] = False
        defaults["useTrainCorr"] = True
        defaults["useTestCorr"] = True
        defaults["corrMode"] = "full"  # use 'full' if useCompl=False

        defaults["corrTrainSteps"] = 20  # 80
        defaults["corrTestMaxSteps"] = 20  # 40 # 60

        defaults["corrEps"] = 1e-2  # antes era 1e-3
        defaults["corrLr"] = (
            0.5  # anterior era 0.1         # use 1e-5 if useCompl=False
        )

        defaults["corrMomentum"] = 0.01

        defaults["corrLrStartPart"] = 1e-1
        defaults["corrLrEndPart"] = 1e-3
        defaults["corrMomentumStartPart"] = 0.5
        defaults["corrMomentumEndPart"] = 0.5
        defaults["qtySamples"] = 50

    elif "nonlinear_2ineq" in prob_type:
        defaults["epochs"] = 100
        defaults["batchSize"] = 200
        defaults["lr"] = 1e-3
        defaults["hiddenSize"] = 200
        defaults["softWeight"] = 10  # use 100 if useCompl=False
        defaults["softWeightEqFrac"] = 0.5
        defaults["useCompl"] = True
        defaults["useTrainCorr"] = True
        defaults["useTestCorr"] = True
        defaults["corrMode"] = "partial"  # use 'full' if useCompl=False
        defaults["corrTrainSteps"] = 5
        defaults["corrTestMaxSteps"] = 5
        defaults["corrEps"] = 1e-4
        defaults["corrLr"] = 1e-4
        defaults["corrLrStart"] = 0
        defaults["corrLrDuration"] = 0
        defaults["corrMomentum"] = 0.5
        defaults["qtySamples"] = 2

    else:
        raise NotImplementedError

    return defaults
