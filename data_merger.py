import pickle
import gzip
import os
import numpy as np


def check_datasets():
    datasets = os.listdir("./trajectories/")
    # datasets = [dataset for dataset in datasets if dataset.endswith(".pkl.*")]
    for dataset_path in datasets:
        
        # print(f"Loading dataset from {dataset_path}")
        dataset_path = f"./trajectories/{dataset_path}"
        env_size = dataset_path.split('/')[-1].split('_')[3]
        dataset_type = dataset_path.split('/')[-1].split('_')[4]        
        if dataset_type == "mixed":
            dataset_type = dataset_type + "_" + dataset_path.split('/')[-1].split('_')[5]
            dataset_type = dataset_type + "_" + dataset_path.split('/')[-1].split('_')[6]
            
        # if "250" not in dataset_path:            
        #     continue
        print('=' * 50)
        print(f"Environment {env_size} CS | Dataset type: {dataset_type}")

        if "gz" in dataset_path:
            with gzip.open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)
        else:
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)

        traj_lens, returns = [], []
        for path in trajectories:
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens = np.array(traj_lens)
        num_timesteps = sum(traj_lens)

        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(
            f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)


def merge_datasets():
    datasets = [
        'PST_V2G_ProfixMax_250_optimal_250_1000.pkl.gz',
        'PST_V2G_ProfixMax_250_optimal_250_1500.pkl.gz',
        'PST_V2G_ProfixMax_250_optimal_250_3500.pkl.gz',
        'PST_V2G_ProfixMax_250_optimal_250_5000.pkl.gz',
        'PST_V2G_ProfixMax_250_optimal_250_5500.pkl.gz',       
    ]
    final_dataset = []
        
    for i, dataset_path in enumerate(datasets):
        print('=' * 50)
        # print(f"Loading dataset from {dataset_path}")
        dataset_path = f"./trajectories/{dataset_path}"
        env_size = dataset_path.split('/')[-1].split('_')[3]
        dataset_type = dataset_path.split('/')[-1].split('_')[4]

        print(f"Environment {env_size} CS | Dataset type: {dataset_type}")
        if "gz" in dataset_path:
            with gzip.open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)
        else:
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)

        traj_lens, returns = [], []
        for path in trajectories:
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens = np.array(traj_lens)
        num_timesteps = sum(traj_lens)
        
        if i == 2:
            best_trajectories_1000 = trajectories[:1000]
            best_trajectories_100 = trajectories[:100]

        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(
            f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)
        
        final_dataset.extend(trajectories)
        
        
        
    #select 10_000 trajectories from the final dataset and save as pkl.gz
    # final_dataset = final_dataset[:10000]    
        
    print('=' * 50)
    print(f"Final dataset")
    traj_lens, returns = [], []
    for path in final_dataset:
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens = np.array(traj_lens)
    num_timesteps = sum(traj_lens)
    
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(  
        f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    
    final_dataset = final_dataset[:3000]    
        
    print('=' * 50)
    print(f"Final dataset")
    traj_lens, returns = [], []
    for path in final_dataset:
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens = np.array(traj_lens)
    num_timesteps = sum(traj_lens)
    
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(  
        f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    
    # print(f"Final dataset 1000")
    # traj_lens, returns = [], []
    # for path in best_trajectories_1000:
    #     traj_lens.append(len(path['observations']))
    #     returns.append(path['rewards'].sum())
    # traj_lens = np.array(traj_lens)
    # num_timesteps = sum(traj_lens)
    
    # print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    # print(  
    #     f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    # print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    # print('=' * 50)
    
    # print(f"Final dataset 100")
    # traj_lens, returns = [], []
    # for path in best_trajectories_100:
    #     traj_lens.append(len(path['observations']))
    #     returns.append(path['rewards'].sum())
    # traj_lens = np.array(traj_lens)
    # num_timesteps = sum(traj_lens)
    
    # print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    # print(  
    #     f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    # print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    # print('=' * 50)
    
    
    

    with gzip.open(f"./trajectories/PST_V2G_ProfixMax_250_optimal_250_3000.pkl.gz", 'wb') as f:
        pickle.dump(final_dataset, f)
    
    # with gzip.open(f"./trajectories/PST_V2G_ProfixMax_25_optimal_25_1000.pkl.gz", 'wb') as f:
    #     pickle.dump(best_trajectories_1000, f)
        
    # with gzip.open(f"./trajectories/PST_V2G_ProfixMax_25_optimal_25_100.pkl.gz", 'wb') as f:
    #     pickle.dump(best_trajectories_100, f)

def create_mixed_dataset():
    
    # create dataset consisting of 1000 trajectories x% optimal and 100-x% random
    opt_dataset_path = f"./trajectories/PST_V2G_ProfixMax_50_PES_optimal_50_5000.pkl.gz"
    with gzip.open(opt_dataset_path, 'rb') as f:
        opt_trajectories = pickle.load(f)
    
    random_dataset_path = f"./trajectories/PST_V2G_ProfixMax_50_PES_random_50_5000.pkl.gz"
    with gzip.open(random_dataset_path, 'rb') as f:
        random_trajectories = pickle.load(f)
    
    bau_dataset_path = f"./trajectories/PST_V2G_ProfixMax_50_PES_bau_50_5000.pkl.gz"
    with gzip.open(bau_dataset_path, 'rb') as f:
        bau_trajectories = pickle.load(f)
        
    mixed_dataset_50 = opt_trajectories[:2500] + random_trajectories[:2500]    
    # mixed_dataset_75 = opt_trajectories[:750] + random_trajectories[:250]
    # mixed_dataset_25 = opt_trajectories[:250] + random_trajectories[:750]
    

    # with gzip.open(f"./trajectories/PST_V2G_ProfixMax_50_PES_mixed_optimal_random_5000.pkl.gz", 'wb') as f:
    #     pickle.dump(mixed_dataset_50, f)
    
    with gzip.open(f"./trajectories/PST_V2G_ProfixMax_50_PES_mixed_optimal_bau_5000.pkl.gz", 'wb') as f:
        pickle.dump(opt_trajectories[:2500] + bau_trajectories[:2500], f)
    
    with gzip.open(f"./trajectories/PST_V2G_ProfixMax_50_PES_mixed_optimal_random_bau_5000.pkl.gz", 'wb') as f:
        pickle.dump(opt_trajectories[:2500] + random_trajectories[:1250] + bau_trajectories[:1250], f)
    
    # with gzip.open(f"./trajectories/PST_V2G_ProfixMax_25_mixed_opt_75_25_1000.pkl.gz", 'wb') as f:
    #     pickle.dump(mixed_dataset_75, f)
        
    # with gzip.open(f"./trajectories/PST_V2G_ProfixMax_25_mixed_opt_25_25_1000.pkl.gz", 'wb') as f:
    #     pickle.dump(mixed_dataset_25, f)

def chop_dataset(b=3000):
    
    dataset_path = f"./trajectories/PST_V2G_ProfixMax_250_random_250_10000.pkl.gz"
    with gzip.open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    trajectories = trajectories[:b]
    traj_lens, returns = [], []
    for path in trajectories:
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens = np.array(traj_lens)
    num_timesteps = sum(traj_lens)
    
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(  
        f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    
    with gzip.open(f"./trajectories/PST_V2G_ProfixMax_250_random_250_{b}.pkl.gz", 'wb') as f:
        pickle.dump(trajectories, f)
    
if __name__ == "__main__":
    # check_datasets()
    # chop_dataset()
    create_mixed_dataset()
    # merge_datasets()
