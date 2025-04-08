def method_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 2
    defaults['simpleIneq'] = 1
    defaults['simpleEq'] = 1
    defaults['simpleEx'] = 1000
    #defaults['nonconvexVar'] = 100
    #defaults['nonconvexIneq'] = 50
    #defaults['nonconvexEq'] = 50
    #defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50


    if prob_type == 'nonlinear':
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
    elif 'dc_wss' in prob_type:
        defaults['epochs'] = 5
        defaults['batchSize'] = 200
        defaults['lr'] = 1e-3
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10           # use 100 if useCompl=False
        defaults['softWeightEqFrac'] = 0.5
        defaults['useCompl'] = False
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'full'    # use 'full' if useCompl=False
        defaults['corrTrainSteps'] = 5
        defaults['corrTestMaxSteps'] = 5
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-5           # use 1e-5 if useCompl=False
        defaults['corrMomentum'] = 0.5 
 
 
#    elif 'dc_wss' in prob_type:
#        defaults['epochs'] = 50   # Aumentar o número de épocas para permitir melhor convergência
#        defaults['batchSize'] = 128  # Reduzir o batch size para suavizar as atualizações de gradiente
#        defaults['lr'] = 5e-4  # Diminuir a taxa de aprendizado para reduzir oscilações
#        defaults['hiddenSize'] = 256  # Aumentar a capacidade da rede para lidar melhor com restrições
#        defaults['softWeight'] = 1.0  # Aumentar o peso das restrições para priorizar viabilidade
#        defaults['softWeightEqFrac'] = 0.3  # Diminuir a penalização das restrições de igualdade
#        defaults['useCompl'] = False
#        defaults['useTrainCorr'] = True
#        defaults['useTestCorr'] = True
#        defaults['corrMode'] = 'full'  # Usar correção parcial para maior estabilidade
#        defaults['corrTrainSteps'] = 5  # Aumentar os passos de correção para refinar a solução
#        defaults['corrTestMaxSteps'] = 5  
#        defaults['corrEps'] = 1e-5  # Reduzir tolerância para garantir soluções mais precisas
#        defaults['corrLr'] = 5e-6  # Diminuir a taxa de aprendizado da correção para evitar saltos bruscos
#        defaults['corrMomentum'] = 0.7  # Aumentar o momentum para suavizar atualizações
#   
   
       
    else:
        raise NotImplementedError

    return defaults