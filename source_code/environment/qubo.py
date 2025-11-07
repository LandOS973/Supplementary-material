from utils.walsh_expansion import WalshExpansion
from tqdm import tqdm
import os
import torch
from random import sample
import numpy as np



def get_Score_trajectoriesQUBO_cuda(strategy, N, nb_instances, nb_restarts, budget, size_pop, tensor_Q, device, verbose , name_file):

    size_pop = strategy.lambda_

    # tensor_Q is expected to have shape (total_cases, N, N) where total_cases = nb_instances_found * nb_restarts
    # repeat to match population size
    tensor_Q = (tensor_Q.unsqueeze(1)).repeat([1, size_pop, 1, 1]).to(device)

    total_cases = tensor_Q.size(0)

    # Now initialize strategy and tracking tensors based on actual available cases
    strategy.reset_learned_parameters(total_cases)
    bestScore = torch.ones(total_cases).to(device) * (-99999)

    nb_iterations = budget // size_pop

    if(verbose):
        pbar = tqdm(range(nb_iterations))
    else:
        pbar = range(nb_iterations)
        
        
    list_tensor_solution = []


    if(name_file is not None):
        f_results = open(name_file, "w")
        f_results.write("runtime, mean, median, std, 2%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 98%" + "\n")
        f_results.close()
    

    
    
    
    


    
    for epoch in pbar:

        tensor_solution = strategy.sample_solutions()

        if epoch == 0:
            startSolution = tensor_solution[:,0,:,:].squeeze(2)
        


        tensor_QUBO = tensor_solution*2 - 1

        Qx = tensor_Q @ tensor_QUBO

        tensor_score = -(torch.transpose(Qx, 2, 3) @ tensor_QUBO).squeeze(2).squeeze(2)  
        

        current_score = torch.max(tensor_score, dim=1).values





        list_tensor_solution.append(tensor_solution)
        
        index_solution = torch.argmax(tensor_score, dim=1)
        index_solution = index_solution.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1,N,1)
        best_current_solution = torch.gather(tensor_solution, 1 , index_solution).squeeze(3).squeeze(1)

        if(epoch == 0):
            bestGlobalSolution = best_current_solution
        else:
            tmp_current_score = current_score.unsqueeze(1).repeat(1,N)
            tmp_bestScore = bestScore.unsqueeze(1).repeat(1,N)
            bestGlobalSolution = torch.where(tmp_current_score > tmp_bestScore, best_current_solution,  bestGlobalSolution)

            
            
        bestScore = torch.where(current_score > bestScore, current_score,  bestScore)
        strategy.updateDistribution( tensor_solution, tensor_score)

        if(verbose):
           pbar.set_postfix(bestScore=torch.mean(bestScore).item(),
                           current_score = torch.mean(current_score).item())


        if(name_file is not None):
            if(((epoch +1)*size_pop) % 100 == 0):
                
                bestScore_np = -bestScore.cpu().numpy()               
                mean = np.mean(bestScore_np)
                median = np.percentile(bestScore_np, 50)
                std = np.std(bestScore_np)
                _2per = np.percentile(bestScore_np, 2)
                _5per = np.percentile(bestScore_np, 5)
                _10per = np.percentile(bestScore_np, 10)
                _25per = np.percentile(bestScore_np, 25)
                _75per = np.percentile(bestScore_np, 75)
                _90per = np.percentile(bestScore_np, 90)
                _95per = np.percentile(bestScore_np, 95)
                _98per = np.percentile(bestScore_np, 98)
                
                f_results = open(name_file, "a")
                f_results.write(str((epoch + 1)*size_pop) + "," +  str(mean) + "," +  str(median) + "," +  str(std) + "," +  str(_2per) + "," +  str(_5per) + "," +  str(_10per) + "," +  str(_25per) + "," +  str(median) + "," +  str(_75per) + "," +  str(_90per) + "," +  str(_95per) + "," +  str(_98per) + "\n")
                f_results.close()



    if(name_file is not None):
        f_hamming = open(name_file + "_HD", "w")
        #f_hamming = open("hamming_distance.csv", "w")
        f_hamming.write("runtime, avg distance, avg std pop" + "\n")
        f_hamming.close()
        
    bestGlobalSolution = bestGlobalSolution.unsqueeze(1).repeat(1,10,1)
    

    
    #startSolution = startSolution.unsqueeze(1).repeat(1,10,1)
    
    for idx, tensor_solution in enumerate(list_tensor_solution):
        
        #if(((idx +1)*size_pop) % 100 == 0):
        
        hamming_distance = torch.sum(torch.abs(tensor_solution.squeeze(3) - bestGlobalSolution), dim=2).cpu().numpy()
            
        #hamming_distance = torch.sum(torch.abs(tensor_solution.squeeze(3) - startSolution), dim=2).cpu().numpy()
        
        
        avg_distance = np.mean(hamming_distance)
        avg_std_distance = np.mean(np.std(hamming_distance, axis = 1))

        f_hamming = open(name_file + "_HD", "a")
        f_hamming.write(str((idx + 1)*size_pop) + "," +  str(avg_distance) + "," +  str(avg_std_distance) + "\n")
        f_hamming.close()
                    
    
    return bestScore_np




def getTensorInstances_QUBO(path, nb_instances, nb_restarts,  N, t, device, phase):

    list_matrix_Q = []
    list_matrix_K = []
    # Ensure path exists and discover available instance files matching pattern
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instances path not found: {path}")

    prefix = f"puboi_evo_n_{N}_t_{t}_i_"
    files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith('.json')]

    if len(files) == 0:
        raise FileNotFoundError(f"No QUBO instance files found in {path} with prefix {prefix}")

    # extract instance numbers and sort
    def inst_index(fname):
        try:
            part = fname[len(prefix):-5]  # strip prefix and .json
            return int(part)
        except Exception:
            return 0

    files_sorted = sorted(files, key=inst_index)

    # select up to nb_instances available files
    selected_files = files_sorted[:nb_instances]

    if len(selected_files) < nb_instances:
        print(f"Warning: requested {nb_instances} instances but only found {len(selected_files)} in {path}. Using {len(selected_files)} instances.")

    for fname in selected_files:
        filename = os.path.join(path, fname)
        f = WalshExpansion()
        f.load(filename)
        Q = f.to_symmetric_Q()

        Q_th = torch.tensor(Q, dtype=torch.float32)

        for i in range(nb_restarts):
            list_matrix_Q.append(Q_th)


    with torch.no_grad():

        tensor_Q = torch.stack(list_matrix_Q, dim=0)


    return tensor_Q
