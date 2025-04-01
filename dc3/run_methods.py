import subprocess
from tqdm import tqdm

def main():

    cases = {
        # 100 epocas
        ('dc_wss_dataset_dc5_ex10',100),
        ('dc_wss_dataset_dc5_ex20',100),
        ('dc_wss_dataset_dc5_ex30',100),
        ('dc_wss_dataset_dc5_ex50',100),
        ('dc_wss_dataset_dc5_ex100',100),
        ('dc_wss_dataset_dc5_ex200',100),
        ('dc_wss_dataset_dc5_ex500',100),
        ('dc_wss_dataset_dc5_ex1000',100),        
        
        # 500 epocas
        ('dc_wss_dataset_dc5_ex10',500),
        ('dc_wss_dataset_dc5_ex20',500),
        ('dc_wss_dataset_dc5_ex30',500),
        ('dc_wss_dataset_dc5_ex50',500),
        ('dc_wss_dataset_dc5_ex100',500),
        ('dc_wss_dataset_dc5_ex200',500),
        ('dc_wss_dataset_dc5_ex500',500),
        ('dc_wss_dataset_dc5_ex1000',500),
        
        # 1000 epocas
        ('dc_wss_dataset_dc5_ex10',1000),
        ('dc_wss_dataset_dc5_ex20',1000),
        ('dc_wss_dataset_dc5_ex30',1000),
        ('dc_wss_dataset_dc5_ex50',1000),
        ('dc_wss_dataset_dc5_ex100',1000),
        ('dc_wss_dataset_dc5_ex200',1000),
        ('dc_wss_dataset_dc5_ex500',1000),
        ('dc_wss_dataset_dc5_ex1000',1000)
    }

    for filename, epochs in tqdm(cases):
        # Executa o script Python com os argumentos desejados
        subprocess.run(['python', 'method.py', '--fileName', filename, '--epochs', str(epochs)])
        
        
if __name__ == "__main__":
    main()
# python methods.py --probType 'dc_wss' --fileName 'dc_wss_dataset_dc5_ex10' --epochs 100