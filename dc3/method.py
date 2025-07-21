from functools import reduce
import operator
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse

from tqdm import tqdm
from utils import str_to_bool

import utils as utils

import default_args

import matplotlib.pyplot as plt

from plot_nonlinear_evolution import plot_nonlinear_evolution
from plot_nonlinear_2ineq_evolution import plot_nonlinear_2ineq_evolution
from plot_nivel_tanque_new import plot_nivel_tanque_new
from plot_simple import plot_simple

import warnings
import datetime


torch.set_printoptions(precision=4, sci_mode=False)

warnings.filterwarnings("ignore")

COMPUTER_RUN = 'personal'  # 'personal' or 'server'

SAVE_PLOT_Y_NEW = False
SAVE_PLOT_GIF = False
SAVE_PLOT_COST = False # True
QTY_EPOCH_SAVE = 40
FIXED_Y_VALUE = False
VALID_PLOT_COST = False
TRAIN_PLOT_COST = False
TEST_PLOT_COST = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device('xpu' if torch.xpu.is_available() else DEVICE)
        
print('Device: ', DEVICE)

def main():
    parser = argparse.ArgumentParser(description='DC3')
    parser.add_argument('--probType', type=str, default='dc_wss',
        choices=['nonlinear', 'nonlinear_2ineq', 'dc_wss'], help='problem type')
    parser.add_argument('--simpleVar', type=int, 
        help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int,
        help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int,
        help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int,
        help='total number of datapoints for simple problem')
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
    parser.add_argument('--corrLrStart', type=float,
        help='starting learning rate for correction procedure')
    parser.add_argument('--corrLrDuration', type=float,
        help='duration of learning rate for correction procedure')
    parser.add_argument('--corrMomentum', type=float,
        help='momentum for correction procedure')
    parser.add_argument('--saveAllStats', type=str_to_bool,
        help='whether to save all stats, or just those from latest epoch')
    parser.add_argument('--resultsSaveFreq', type=int,
        help='how frequently (in terms of number of epochs) to save stats to file')
    parser.add_argument('--dc', type=int, default=5,
        help='number of duty cycles')
    parser.add_argument('--qtySamples', type=int, default=30)
    parser.add_argument('--fileName', type=str, default=None)   
    parser.add_argument('--epochs', type=int,
        help='number of neural network epochs')
    parser.add_argument('--softWeightEqFracStart', type=float,
        help='starting value of softWeightEqFrac')
    parser.add_argument('--softWeightEqFracDuration', type=float, 
        help='duration of softWeightEqFrac')
     
    args = parser.parse_args()
    args = vars(args) # change to dictionary
        
    if args['fileName'] is None:
            args['fileName'] = f"dc_wss_dataset_dc{args['dc']}_ex{args['qtySamples']}"
    
    defaults = default_args.method_default_args(args['probType'])
    
    
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    
    print(args)

    setproctitle('DC3-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'nonlinear':
        filepath = os.path.join('datasets', 'nonlinear', "random_nonlinear_dataset_ex{}".format(args['simpleEx']))      
    elif prob_type == 'nonlinear_2ineq':
        filepath = os.path.join('datasets', 'nonlinear_2ineq', "random_nonlinear_2ineq_dataset_ex{}".format(args['simpleEx']))
    elif prob_type == 'dc_wss':
        filepath = os.path.join('datasets', 'dc_wss',  f"dc_wss_dataset_dc{args['dc']}_ex{args['qtySamples']}")
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
        
    time_trainning_start = time.time()
        
    solver_step = args['lr']
    nepochs = args['epochs']
    batch_size = args['batchSize']

    train_dataset = TensorDataset(data.trainX)
    valid_dataset = TensorDataset(data.validX)
    test_dataset = TensorDataset(data.testX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    solver_net = NNSolver(data, args)
    solver_net.to(DEVICE)
    solver_opt = optim.Adam(solver_net.parameters(), lr=solver_step)
    
    stats = {}
    train_losses = []
    valid_losses = []
    test_losses = []

    valid_eval_values = []
    test_eval_values = []
    
    y1_new_history = []
    y2_new_history = [] 

    
    for i in tqdm(range(nepochs)):
        epoch_stats = {}

        # Get valid loss
        solver_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_net(data, Xvalid, solver_net, args, 'valid', epoch_stats)
                                    
        solver_net.eval()
        for Xtest in test_loader:
            Xtest = Xtest[0].to(DEVICE)
            eval_net(data, Xtest, solver_net, args, 'test', epoch_stats)
            
        solver_net.train()
        for Xtrain in train_loader:    
                           
            Xtrain = Xtrain[0].to(DEVICE)
            start_time = time.time()
            

########################################################################
            
            #print('Begin Y hat')
            #y_hat_start_time = time.time()
            Yhat_train = solver_net(Xtrain) # 1. Forward pass
            
            #print('Time Y Yhat_train ', time.strftime("%H:%M:%S", time.gmtime(time.time() - y_hat_start_time)))
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            if SAVE_PLOT_Y_NEW:

                for j in range(Yhat_train.shape[0]):
                    
                    plot_simple(Yhat_train[j].cpu().detach().numpy(), 000, args, 0, n_sample=j, title_comment='Initial Y New')
                
########################################################################
            
            # UTILIZADO PARA TREINAR UM Y ESPECÍFICO
            if FIXED_Y_VALUE:
                
                Yhat_train = torch.tensor(
                    [
                                                
                     [1.37, 4.27, 12.63, 17.19, 20.17,  2.85, 4.13,  4.51, 2.92,  3.63],

                     [13.9925, 17.8876, 18.6809, 21.6833, 22.9596,  3.5851,  0.7832,  2.8925, 1.1663,  0.9404],

                     [13.9925, 17.8876, 18.6809, 21.6833, 22.9596,  3.5851,  0.7832,  2.8925, 1.1663,  0.9404],

                     [13.9925, 17.8876, 18.6809, 21.6833, 22.9596,  3.5851,  0.7832,  2.8925, 1.1663,  0.9404],

                     [13.9925, 17.8876, 18.6809, 21.6833, 22.9596,  3.5851,  0.7832,  2.8925, 1.1663,  0.9404],

                     [13.9925, 17.8876, 18.6809, 21.6833, 22.9596,  3.5851,  0.7832,  2.8925, 1.1663,  0.9404],

                     #[13.9925, 17.8876, 18.6809, 21.6833, 22.9596,  3.5851,  0.7832,  2.8925, 1.1663,  0.9404],

#                     [13.9925, 17.8876, 18.6809, 21.6833, 22.9596,  3.5851,  0.7832,  2.8925, 1.1663,  0.9404]

                    ]
                ).requires_grad_(True)  
                
            #Y_FIXED_VALUES = False          

########################################################################            

            Ynew_train = grad_steps(data, Xtrain, Yhat_train, args, epoch=i) # 1. Forward pass                             
            # Generate GIF
            if SAVE_PLOT_GIF:
                        
                time_generate_gif = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                if COMPUTER_RUN == 'personal':
                    base_folder = r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots"
                else:
                    base_folder = r"C:\Users\marcostulio\Desktop\dc3\plots"
                
                output_folder = os.path.join(base_folder, f"gifs_epoca_{time_generate_gif}")
                os.makedirs(output_folder, exist_ok=True)            
                

                # Gera GIFs individuais
                for nr_sample in range(Ynew_train.shape[0]):
                    utils.create_gif_from_plots(nr=nr_sample, base_folder=base_folder, output_folder=output_folder)

                # Gera GIF combinado
                utils.create_combined_gif(Y_shape_0=Ynew_train.shape[0], base_folder=base_folder, output_folder=output_folder)
            
########################################################################            

            train_loss = total_loss(data, Xtrain, Ynew_train, args, i) # 2. Calculate de loss   
            
            
            
            
            
            
            if SAVE_PLOT_COST and TRAIN_PLOT_COST:                          
                for j in range(len(train_loss)):
                    y_val = Ynew_train[j].cpu().detach().numpy()
                    levels = data.gT_Original(Ynew_train[j].unsqueeze(0))[0][:-1].cpu().detach().numpy()
                    cost = float(data.obj_fn_Autograd(Ynew_train[j].unsqueeze(0), args)[0])
                    penalty = float(train_loss[j])
                    respected = ''
                    if round(cost, 3) == round(penalty, 3):
                        respected = " - Respected all hard constraints"
                    title = (
                        f"Sample: {j}\n"
                        f"Epoch: {i+1} \n"                        
                        f"Y new: {np.array2string(y_val, precision=2, separator=', ')}\n"
                        f"Levels: {np.array2string(levels, precision=2, separator=', ')}\n"
                        f"Cost: € {cost:.2f} | Cost penalty: € {penalty:.2f}{respected}"
                    )
                    plot_nivel_tanque_new(args, Ynew_train[j], train_loss[j], save_plot=True, title=title, sample=j)
                            
            
########################################################################
            # OBTENHO A FUNCAO DE PERDA
            ##### DEMORA MAIOR ESTÁ NESTE TRECHO #####
            #print('Begin loss backward')
            #time_start_backward = time.time()    
            solver_opt.zero_grad() # 0. Optimizer zero grad
             # 3. Performe backpropagation on the loss with respect to
            train_loss.sum().backward() 
            #print('Time backward ', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start_backward)))
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')



########################################################################

            # USO O GRADIENTE PARA OTIMIZAR
            #time_start_solveropt = time.time()
            #print('Begin solver_opt')
            solver_opt.step() # 4. Performe gradiente descent            
            #print('Time solver_opt ', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start_solveropt)))
            #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

########################################################################          
   
            train_time = time.time() - start_time
            dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            dict_agg(epoch_stats, 'train_time', train_time, op='sum')


            for name, param in solver_net.named_parameters():
                
                ######################################
                #print(f"{name}: {param.grad}")
                pass                                    




                    
        # Média da loss durante a época
        avg_train_loss = np.mean(epoch_stats['train_loss'])
        avg_valid_loss = np.mean(epoch_stats['valid_loss'])
        avg_test_loss = np.mean(epoch_stats['test_loss'])


        avg_valid_eval = np.mean(epoch_stats['valid_eval'])
        avg_test_eval = np.mean(epoch_stats['test_eval'])
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        test_losses.append(avg_test_loss)        

        valid_eval_values.append(avg_valid_eval)
        test_eval_values.append(avg_test_eval)


        print(f"[Epoch {i+1}] Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Time Elapsed: {np.mean(epoch_stats['valid_time'])}")

        
        if args['probType'] == 'dc_wss':
        
            #plot_nivel_tanque_new(args, Ynew_train[0].cpu().detach().numpy(), data.obj_fn_Autograd(Ynew_train, args)[0].cpu().detach().numpy(), save_plot=True, show=False)

            pass
            
        print(
            'Epoch {}: train loss {:.4f}, eval {:.4f}, dist {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, steps {}, time {:.4f}'.format(
                i, np.mean(epoch_stats['train_loss']), np.mean(epoch_stats['valid_eval']),
                np.mean(epoch_stats['valid_dist']), np.mean(epoch_stats['valid_ineq_max']),
                np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
                np.mean(epoch_stats['valid_steps']), np.mean(epoch_stats['valid_time'])
            )
        )
        #print('Y new [0]: ', Ynew_train[0].cpu().detach().numpy())
        print('----')
        print('')
        print('')
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

    # Gráfico das perdas
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Valid')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('plots',f'loss_curve_{args['simpleEx']}_epochs_{args['epochs']}_{now}.png'))
    plt.show()

    # Gráfico da função objetivo (eval)
    plt.figure()
    plt.plot(valid_eval_values, label='Eval - Valid')
    plt.plot(test_eval_values, label='Eval - Test')
    plt.xlabel('Epoch')
    plt.ylabel('Obj fn')
    plt.title('Evolution obj fn (eval)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('plots',f'eval_curve_{args['simpleEx']}_epochs_{args['epochs']}_{now}.png'))
    plt.show()


    if args['probType'] == 'nonlinear':
        plot_nonlinear_evolution(data, y1_new_history, y2_new_history, os.path.join('plots',f'plot_{now}_ex_{args['simpleEx']}_epochs_{args['epochs']}.png'))

    if args['probType'] == 'nonlinear_2ineq':

        plot_nonlinear_2ineq_evolution(data, y1_new_history, y2_new_history, os.path.join('plots',f'plot_{now}_ex_{args['simpleEx']}_epochs_{args['epochs']}.png'))
        
    
    
    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
        
    # A SALVAR O MODELO
#    with open(os.path.join(save_dir, f'solver_net_{now}_dc{args['dc']}_samples{args['qtySamples']}_epochs{args["epochs"]}.dict'), 'wb') as f:
#        torch.save(solver_net.state_dict(), f)
        
        if args['probType'] == 'nonlinear' or args['probType'] == 'nonlinear_2ineq':
        
            torch.save(f'model_{now}_{args['simpleEx']}_epochs_{args['epochs']}.pt', f)
    
    save_model = os.path.join('models')
    
    with open(os.path.join(save_model, f'model_{now}_dc{args['dc']}_samples{args['qtySamples']}_epochs{args["epochs"]}.pt'), 'wb') as f:
         torch.save(solver_net.state_dict(), f)


    print('----')    
    print('BENCHMARK FONTINHAOPTIMIZATION: 110.65')
    print('----')    
    print('Training finished')   


    print('Elapsed time: ', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_trainning_start)))

    if args['probType'] == 'dc_wss':

        # Testar resultado do modelo
        path_model = os.path.join(save_model, f'model_{now}_dc{args['dc']}_samples{args['qtySamples']}_epochs{args["epochs"]}.pt')
        
        hiddenSize_ = args['hiddenSize']
        args_ = {'probType': 'dc_wss', 'hiddenSize': hiddenSize_, 'useCompl': False, 'corrMode': 'full'}

        newModel = NNSolver(data, args_)
        newModel.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
        newModel.eval()    

        if args['dc'] == 3:
            input_data = torch.tensor([[1,8,12,3,3,3.0]])
            #input_data = np.array(input_data, dtype=np.float32)
            #input_data = [1,8,12,3,3,3.0]
        elif args['dc'] == 4:
            input_data = torch.tensor([[1,8,12,18,3,3,3.0,2.5]])
            #input_data = np.array(input_data, dtype=np.float32)
        elif args['dc'] == 5:
            input_data = torch.tensor([[0, 4.8, 9.6, 14.4, 19.2, 1, 1, 1, 1, 1]])
            #input_data = np.array(input_data, dtype=np.float32)
            
        output_data = newModel(input_data)
        
        if args['probType'] == 'dc_wss':            
            total_cost = data.obj_fn_Autograd(output_data, args)[0]
        else:
            total_cost = data.obj_fn_Original(output_data, args)[0]
        
        
        print('#####')
        print('Input: ', input_data)
        print('#####')
        print('Output: ', output_data)    
        print('#####')
        print('gT: ', data.gT_Original(output_data[0].unsqueeze(0)) )
        print('#####')
        print('Resultado: ',total_cost )
        print('Avaliation finished')


        plot_nivel_tanque_new(args, output_data[0].cpu().detach().numpy(), total_cost, save_plot=True)

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
    
    
    if SAVE_PLOT_COST and TEST_PLOT_COST and args['probType'] == 'dc_wss':    
        
        
        for i in range(len(Ynew)):
            
            y_val = Ynew[i].cpu().detach().numpy()
            levels = data.gT_Original(Ynew[i].unsqueeze(0))[0][:-1].cpu().detach().numpy()
            cost = float(data.obj_fn_Autograd(Ynew[i].unsqueeze(0), args)[0])
            
            title = (
                f"Y new from {prefix.upper()}\n"
                f"Sample: {i}\n"
                f"Y new: {np.array2string(y_val, precision=2, separator=', ')}\n"
                f"Levels: {np.array2string(levels, precision=2, separator=', ')}\n"
                f"Cost: € {cost:.2f}"
            
            )
            
            plot_nivel_tanque_new(args, y_val, cost, save_plot=True, title=title, sample=i)
            
            
    dim = 0 if args['probType'] == 'nonlinear' or args['probType'] == 'nonlinear_2ineq' else 1
    
    dict_agg(stats, make_prefix('time'), end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('steps'), np.array([steps]))
    dict_agg(stats, make_prefix('loss'), total_loss(data, X, Ynew, args, 0).detach().cpu().numpy())
    
    if args['probType'] == 'dc_wss':    
        dict_agg(stats, make_prefix('eval'), data.obj_fn_Autograd(Ycorr, args).detach().cpu().numpy())        
    else:
        dict_agg(stats, make_prefix('eval'), data.obj_fn_Original(Ycorr, args).detach().cpu().numpy())
    
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr, args), dim=dim)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Ycorr, args), dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Ycorr, args) > eps_converge, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Ycorr, args) > 10 * eps_converge, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Ycorr, args) > 100 * eps_converge, dim=dim).detach().cpu().numpy())
    
    
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
    
    if args['probType'] == 'dc_wss':
        dict_agg(stats, make_prefix('raw_eval'), data.obj_fn_Autograd(Ynew, args).detach().cpu().numpy())
    else:
        dict_agg(stats, make_prefix('raw_eval'), data.obj_fn_Original(Ynew, args).detach().cpu().numpy())
    
    dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(data.ineq_dist(X, Ynew, args), dim=dim)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(data.ineq_dist(X, Ynew, args), dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Ynew, args) > eps_converge, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Ynew, args) > 10 * eps_converge, dim=dim).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Ynew, args) > 100 * eps_converge, dim=dim).detach().cpu().numpy())

        
    return stats



def total_loss(data, X, Y, args, i):
    
    dim = 0 if args['probType'] == 'nonlinear' or args['probType'] == 'nonlinear_2ineq' else 1

    if args['probType'] == 'dc_wss':
        obj_cost = data.obj_fn_Autograd(Y, args)
        #obj_cost = data.obj_fn_Original(Y, args)
    else:
        obj_cost = data.obj_fn_Original(Y, args)
    
    if args['probType'] == 'nonlinear' or args['probType'] == 'nonlinear_2ineq':
        ineq_dist = data.ineq_dist(Y, Y, args)
    else:
        ineq_dist = data.ineq_dist(X, Y, args)

    ineq_cost = torch.norm(ineq_dist, dim=1)
        
    if args['probType'] == 'dc_wss':
        
        # Somente com restricao de desigualdade
        #result = obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost
        #result = obj_cost + args['softWeight'] * ineq_cost
        #result = obj_cost
        
        result = obj_cost + 0.5 * ineq_cost
        
        
    else:
        # Com equações de igualdade e desigualdade
        eq_cost = torch.norm(data.eq_resid(X, Y).unsqueeze(1), dim=1)
        result = obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + args['softWeight'] * args['softWeightEqFrac'] * eq_cost
    
    return result



def grad_steps(data, X, Y, args, epoch=None):

        
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
        #print('')                
        #print(f'-- GradSteps -- | Qty {str(Y.shape[0])}')
        for i in range(num_steps+1):            
            if partial_corr:
                Y_step = data.ineq_partial_grad(X, Y_new)
            else:       
                if args['probType'] == 'dc_wss':                                                             
                    ineq_step = data.ineq_grad(X, Y_new, args)   
                                        
                    mid_point = ineq_step.shape[1] // 2
                    
                    new_Y_step_start = ( 1 - args['softWeightEqFracStart']) * ineq_step[:, :mid_point]
                    new_Y_step_end = ( 1 - args['softWeightEqFracDuration']) * ineq_step[:, mid_point:]
                    
                    Y_step = torch.cat([new_Y_step_start, new_Y_step_end], dim=1)
                     
                else:
                    ineq_step = data.ineq_grad(X, Y_new)                
                    eq_step = data.eq_grad(X, Y_new)
                                
                    Y_step = (1 - args['softWeightEqFrac']) * ineq_step + args['softWeightEqFrac'] * eq_step

            
            if i % QTY_EPOCH_SAVE == 0 and len(data.trainX) == len(Y) and SAVE_PLOT_Y_NEW:
                            
                histories = [[] for _ in range(Y.shape[0])]
                #for j in range(len(Y) // 2):
                for j in range(len(Y)):
                    histories[j].append(-Y_step[j].detach().numpy())                                        
            
            new_Y_step = lr * Y_step + momentum * old_Y_step            


            Y_new = Y_new - new_Y_step

            if i % QTY_EPOCH_SAVE == 0 and len(data.trainX) == len(Y) and SAVE_PLOT_Y_NEW:
                
                
                for j in range(Y.shape[0]):
                    
                    if i == args['corrTrainSteps']:
                    
                        plot_simple(results=Y_new[j].cpu().detach().numpy(), iteration=i, args=args, y_steps=histories[j], n_sample=j, title_comment='Final Y New',total_iteration=args['corrTrainSteps'], epoch=epoch)
                    
                    else: 
                        
                        plot_simple(results=Y_new[j].cpu().detach().numpy(), iteration=i, args=args, y_steps=histories[j], n_sample=j, total_iteration=args['corrTrainSteps'], epoch=epoch)
                        
                


            old_Y_step = new_Y_step   
        
        

        # Diretório onde os plots estão salvos
        #plot_dir = "C:\\Users\\mtcd\\Documents\\Codes\\dc3-wss\\dc3\\plots"
        plot_dir = os.path.join('plots')

        # Cria o GIF com os arquivos que começam com "plot_simple_nr0_epochNr"
        
        if i % QTY_EPOCH_SAVE == 0 and len(data.trainX) == len(Y) and SAVE_PLOT_Y_NEW:
            for i in range(Y.shape[0]):
                pass
                #gif_filename = os.path.join('plots', f"training_progress_nr{i}.gif")
                #utils.create_gif_from_plots(plot_dir, gif_filename, prefix=f"plot_simple_nr{i}_epochNr")
                    
        return Y_new
       
    else:
        return Y
    


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
                           torch.max(data.ineq_dist(X, Y_new, args)) > eps_converge) and i < max_steps:
                if partial_corr:
                    Y_step = data.ineq_partial_grad(X, Y_new)
                else:
                    
                    if args['probType'] == 'dc_wss':
                        ineq_step = data.ineq_grad(X, Y_new, args)
                    
                        mid_point = ineq_step.shape[1] // 2
                        
                        new_Y_step_start = ( 1 - args['softWeightEqFracStart']) * ineq_step[:, :mid_point]
                        new_Y_step_end = ( 1 - args['softWeightEqFracDuration']) * ineq_step[:, mid_point:]
                        
                        Y_step = torch.cat([new_Y_step_start, new_Y_step_end], dim=1)

                    else:
                        
                        ineq_step = data.ineq_grad(X, Y_new, args)
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
        x = x.float()
 
        out = self.net(x)
        #print('OUT ', out.requires_grad, out.grad_fn)  # depois da rede
        if self._args['useCompl']:
            result = self._data.complete_partial(x, out)
            return result
        else:
            
            if self._args['probType'] == 'dc_wss':
                
                
                result = self._data.process_output(x, out)            
            
            
            return result   


if __name__=='__main__':
    main()