from functools import reduce
import operator
import torch

from torch import nn
from torch import optim

torch.set_default_dtype(torch.float64)



from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse

from utils import str_to_bool
import default_args

import matplotlib.pyplot as plt

from plot_nonlinear_evolution import plot_nonlinear_evolution

import warnings
import datetime

warnings.filterwarnings("ignore")

import EPANET_API as EPA_API


# RESULTADO DE FONTINHAOPTIMIZATION: 110.65

# to run:
# python .\method.py --probType nonlinear --simpleEx 10000 --simpleVar 2 --simpleEq 1 --simpleIneq 1 --corrMode 'partial' --useCompl True --epoch 150

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
print('Device: ', DEVICE)

def main():
    parser = argparse.ArgumentParser(description='DC3')
    parser.add_argument('--probType', type=str, default='dc_wss',
        choices=['nonlinear', 'dc_wss'], help='problem type')
    parser.add_argument('--simpleVar', type=int, 
        help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int,
        help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int,
        help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int,
        help='total number of datapoints for simple problem')
#    parser.add_argument('--nonconvexVar', type=int,
#        help='number of decision vars for nonconvex problem')
#    parser.add_argument('--nonconvexIneq', type=int,
#        help='number of inequality constraints for nonconvex problem')
#    parser.add_argument('--nonconvexEq', type=int,
#        help='number of equality constraints for nonconvex problem')
#    parser.add_argument('--nonconvexEx', type=int,
#        help='total number of datapoints for nonconvex problem')
    parser.add_argument('--epochs', type=int,
        help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int,
        help='training batch size')
    parser.add_argument('--lr', type=float,
        help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int,
        help='hidden layer size for neural network')
    parser.add_argument('--softWeight', type=float,
        help='total weight given to constraint violations in loss')
    parser.add_argument('--softWeightEqFrac', type=float,
        help='fraction of weight given to equality constraints (vs. inequality constraints) in loss')
    parser.add_argument('--useCompl', type=str_to_bool,
        help='whether to use completion')
    parser.add_argument('--useTrainCorr', type=str_to_bool,
        help='whether to use correction during training')
    parser.add_argument('--useTestCorr', type=str_to_bool,
        help='whether to use correction during testing')
    parser.add_argument('--corrMode', choices=['partial', 'full'],
        help='employ DC3 correction (partial) or naive correction (full)')
    parser.add_argument('--corrTrainSteps', type=int,
        help='number of correction steps during training')
    parser.add_argument('--corrTestMaxSteps', type=int,
        help='max number of correction steps during testing')
    parser.add_argument('--corrEps', type=float,
        help='correction procedure tolerance')
    parser.add_argument('--corrLr', type=float,
        help='learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float,
        help='momentum for correction procedure')
    parser.add_argument('--saveAllStats', type=str_to_bool,
        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
        help='how frequently (in terms of number of epochs) to save stats to file')


#    defaults = {'nonlinear': {'probType': 'nonlinear', 'simpleVar': 2, 'simpleIneq': 1, 'simpleEq': 1, 'simpleEx': 200, 'epochs': 100, 'corrMode': 'partial', 'useCompl': True }}


    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.method_default_args(args['probType'])
    
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    
    print(args)

    setproctitle('DC3-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    #if prob_type == 'simple':
    #    filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
    #        args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    #elif prob_type == 'nonconvex':
    #    filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
    #        args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    if prob_type == 'nonlinear':
        filepath = os.path.join('datasets', 'nonlinear', "random_nonlinear_dataset_ex{}".format(args['simpleEx']))      
    elif prob_type == 'dc_wss':
        filepath = os.path.join('datasets', 'dc_wss', 'dc_wss_dataset_dc_5_ex_500')
    else:
        raise NotImplementedError
    with open(filepath, 'rb') as f:
        data = pickle.load(f)        
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE

    date_format = '%Y-%m-%d %H:%M:%S'
    date = time.strftime(date_format, time.localtime())
    date = date.replace(':', '-').replace(' ', '_')

    # Processa `data` para garantir que o nome seja válido
    data_str = str(data).replace('<', '').replace('>', '').replace(':', '').replace(' ', '_')

    # Cria o caminho para salvar os resultados
    save_dir = os.path.join('results', data_str, 'method', f'result_{date}_epochs_{args["epochs"]}')

    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
    
    # Run method
    train_net(data, args, save_dir)


def train_net(data, args, save_dir):
    
    solver_step = args['lr']
    nepochs = args['epochs']
    batch_size = args['batchSize']


    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)#, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_net = NNSolver(data, args)
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)

    stats = {}
    train_losses = []
    
    y1_new_history = []
    y2_new_history = [] 
        
    final_result = {}
    
    for i in range(nepochs):
        epoch_stats = {}

        solver_net.train()
        for Xtrain in train_loader:
                   
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
            solver_opt.zero_grad() # 0. Optimizer zero grad
            Yhat_train = solver_net(Xtrain) # 1. Forward pass
            Ynew_train = grad_steps(data, Xtrain, Yhat_train, args) #gradiente steps for Y
            train_loss = total_loss(data, Xtrain, Ynew_train, args) # 2. Calculate de loss            
            train_loss.requires_grad = True
            train_loss.sum().backward() # 3. Performe backpropagation on the loss with respect to the parameters of the model
            solver_opt.step() # 4. Performe gradiente descent
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')

        
        # Get valid loss
        solver_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_net(data, Xvalid, solver_net, args, 'valid', epoch_stats)


        solver_net.eval()
        for Xtest in test_loader:
            Xtest = Xtest[0].to(DEVICE)
            eval_net(data, Xtest, solver_net, args, 'test', epoch_stats)

          
        # Média da loss durante a época
        avg_train_loss = np.mean(epoch_stats['train_loss'])
        train_losses.append(avg_train_loss)
                
        print(
            'Epoch {}: train loss {:.4f}, eval {:.4f}, dist {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, steps {}, time {:.4f}, y_new_train[0] {:.4f}, y_new_train[1] {:.4f}'.format(
                i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                np.mean(epoch_stats['valid_dist']), np.mean(epoch_stats['valid_ineq_max']),
                np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
                #np.mean(epoch_stats['valid_eq_max']), 
                np.mean(epoch_stats['valid_steps']), np.mean(epoch_stats['valid_time']),
                np.mean(Ynew_train[0].cpu().detach().numpy()), np.mean(Ynew_train[1].cpu().detach().numpy())
            )
        )
        
        print('RESULTADO DE FONTINHAOPTIMIZATION: 110.65')
        
        y1_new_history.append(np.mean(Ynew_train[0].cpu().detach().numpy()))
        y2_new_history.append(np.mean(Ynew_train[1].cpu().detach().numpy()))

        if args['saveAllStats']:
            if i == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            stats = epoch_stats

        if (i % args['resultsSaveFreq'] == 0):
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
                torch.save(solver_net.state_dict(), f)

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')    
        
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('plots',f'train_loss_ex_{args['simpleEx']}_epochs_{args['epochs']}_{now}.png'))
    plt.show() 

    if args['probType'] == 'nonlinear':

        y_new_history = list(zip(y1_new_history, y2_new_history))

        #plot_nonlinear_evolution(data, y1_new_history, y2_new_history)
        plot_nonlinear_evolution(data, y1_new_history, y2_new_history, os.path.join('plots',f'plot_{now}_ex_{args['simpleEx']}_epochs_{args['epochs']}.png'))
    
    
    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'solver_net.dict'), 'wb') as f:
        torch.save(solver_net.state_dict(), f)
        
        if args['probType'] == 'nonlinear':
        
            torch.save(f'model_{now}_{args['simpleEx']}_epochs_{args['epochs']}.pt', f)
    
        #else:
            
        #    torch.save(f'model_{now}_dcwss_epochs_{args['epochs']}.pt', f)
    
    
    save_model = os.path.join('models')
    with open(os.path.join(save_model, f'model_{now}_dcwss_epochs_{args['epochs']}.pt'), 'wb') as f:
         torch.save(solver_net.state_dict(), f)
    
    
    print('Training finished')    
    
    return solver_net, stats

# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value

# Modifies stats in place
def eval_net(data, X, solver_net, args, prefix, stats):
    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)

    start_time = time.time()
    Y = solver_net(X)
    base_end_time = time.time()

    Ycorr, steps = grad_steps_all(data, X, Y, args)
    end_time = time.time()

    Ynew = grad_steps(data, X, Y, args)
    raw_end_time = time.time()

    dim = 0 if args['probType'] == 'nonlinear' or args['probType'] == 'nonlinear_ex2' else 1
    
    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('steps'), np.array([steps]))
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Ynew, args).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr), dim=dim)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Ycorr), dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Ycorr) > eps_converge, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Ycorr) > 10 * eps_converge, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Ycorr) > 100 * eps_converge, dim=dim).detach().cpu().numpy())
    
    
    if args['probType'] != 'dc_wss':
    
        dict_agg(stats, make_prefix('eq_max'),
                torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=dim)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=dim).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_num_viol_0'),
                torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > eps_converge, dim=dim).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_num_viol_1'),
                torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 10 * eps_converge, dim=dim).detach().cpu().numpy())
        dict_agg(stats, make_prefix('eq_num_viol_2'),
                torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 100 * eps_converge, dim=dim).detach().cpu().numpy())            
        dict_agg(stats, make_prefix('raw_eq_max'),
                torch.max(torch.abs(data.eq_resid(X, Ynew)), dim=dim)[0].detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_eq_mean'),
                torch.mean(torch.abs(data.eq_resid(X, Ynew)), dim=dim).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
                torch.sum(torch.abs(data.eq_resid(X, Ynew)) > eps_converge, dim=dim).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
                torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 10 * eps_converge, dim=dim).detach().cpu().numpy())
        dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
                torch.sum(torch.abs(data.eq_resid(X, Ynew)) > 100 * eps_converge, dim=dim).detach().cpu().numpy())
                
    dict_agg(stats, make_prefix('raw_time'), (raw_end_time-end_time) + (base_end_time-start_time), op='sum')
    dict_agg(stats, make_prefix('raw_eval'), data.obj_fn(Ynew).detach().cpu().numpy())
    
    dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(data.ineq_dist(X, Ynew), dim=dim)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(data.ineq_dist(X, Ynew), dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Ynew) > eps_converge, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Ynew) > 10 * eps_converge, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Ynew) > 100 * eps_converge, dim=dim).detach().cpu().numpy())

        
    return stats



def grad_steps(data, X, Y, args):
    
    take_grad_steps = args['useTrainCorr']
    
    if take_grad_steps:
        lr = args['corrLr']
        num_steps = args['corrTrainSteps']
        momentum = args['corrMomentum']
        partial_var = args['useCompl']
        partial_corr = True if args['corrMode'] == 'partial' else False
        if partial_corr and not partial_var:
            assert False, "Partial correction not available without completion."
        Y_new = Y
        old_Y_step = 0        
        for i in range(num_steps):            
            if partial_corr:
                Y_step = data.ineq_partial_grad(X, Y_new)
            else:       
                if args['probType'] == 'dc_wss': 
                    ineq_step = data.ineq_grad(X, Y_new)     
                    Y_step = (1 - args['softWeightEqFrac']) * ineq_step
                else:
                    ineq_step = data.ineq_grad(X, Y_new)                
                    eq_step = data.eq_grad(X, Y_new)
                                
                    Y_step = (1 - args['softWeightEqFrac']) * ineq_step + args['softWeightEqFrac'] * eq_step
                
            new_Y_step = lr * Y_step + momentum * old_Y_step
            Y_new = Y_new - new_Y_step

            old_Y_step = new_Y_step            

        return Y_new
        
    else:
        return Y
        
def total_loss(data, X, Y, args):

    dim = 0 if args['probType'] == 'nonlinear' or args['probType'] == 'nonlinear_ex2' else 1

    obj_cost = data.obj_fn(Y)
    
    # observar que no codigo usado para o nonlinear, era (Y, Y)
    ineq_dist = data.ineq_dist(X, Y)

    ineq_cost = torch.norm(ineq_dist, dim=1)    
    
    if args['probType'] == 'dc_wss':

    #   Somente com restricao de desigualdade
        result = obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost
    
    else:
    
            # Com equações de igualdade e desigualdade
        eq_cost = torch.norm(data.eq_resid(X, Y).unsqueeze(1), dim=1)
        result = obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + args['softWeight'] * args['softWeightEqFrac'] * eq_cost
    
    return result

# Used only at test time, so let PyTorch avoid building the computational graph
def grad_steps_all(data, X, Y, args):
    take_grad_steps = args['useTestCorr']
    if take_grad_steps:
        lr = args['corrLr']
        eps_converge = args['corrEps']
        max_steps = args['corrTestMaxSteps']
        momentum = args['corrMomentum']
        partial_var = args['useCompl']
        partial_corr = True if args['corrMode'] == 'partial' else False
        if partial_corr and not partial_var:
            assert False, "Partial correction not available without completion."
        Y_new = Y
        i = 0
        old_Y_step = 0
        old_ineq_step = 0
        old_eq_step = 0
        with torch.no_grad():
            
            while (  i == 0 or ( args['probType'] != 'dc_wss' ) and torch.max(torch.abs(data.eq_resid(X, Y_new))) > eps_converge  or
                           torch.max(data.ineq_dist(X, Y_new)) > eps_converge) and i < max_steps:
                if partial_corr:
                    Y_step = data.ineq_partial_grad(X, Y_new)
                else:
                    
                    if args['probType'] == 'dc_wss':
                        ineq_step = data.ineq_grad(X, Y_new)
                        Y_step = (1 - args['softWeightEqFrac']) * ineq_step
                        
                    else:
                        
                            
                    
                    
                        ineq_step = data.ineq_grad(X, Y_new)
                        eq_step = data.eq_grad(X, Y_new)
                        Y_step = (1 - args['softWeightEqFrac']) * ineq_step + args['softWeightEqFrac'] * eq_step
                
                
                
                
                
                new_Y_step = lr * Y_step + momentum * old_Y_step
                Y_new = Y_new - new_Y_step

                old_Y_step = new_Y_step
                i += 1

        return Y_new, i
    else:
        return Y, 0


######### Models

class NNSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        
        output_dim = data.ydim - data.nknowns        

        if self._args['useCompl']:
            layers += [nn.Linear(layer_sizes[-1], output_dim - data.neq)]            
        else:
            layers += [nn.Linear(layer_sizes[-1], output_dim)] 
            
        for layer in layers:
            if type(layer) is nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)

        if self._args['useCompl']:
            result = self._data.complete_partial(x, out)
            return result
        else:
            result = self._data.process_output(x, out)

            return result
        
if __name__=='__main__':
    main()