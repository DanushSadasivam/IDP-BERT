import torch
import wandb
from datetime import datetime
import yaml
import os
import shutil
from data.dataloader import load_data
from model.network import create_model, setup_training
from model.utils import train, validate, test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')


def train_model():
    best_r2 = 0
    for epoch in range(config['epochs']):
        train_loss, train_r2 = train(model, train_loader, optimizer, criterion, device)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss:.3e}\tTraining R2: {train_r2:.3f}\tLR: {curr_lr:.4e}')

        val_loss, val_r2 = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss:.3e}\tValidation R2: {val_r2:.3f}\n')

        scheduler.step(val_r2)
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss,
                'train_r2': train_r2,
                'val_loss': val_loss,
                'val_r2': val_r2,
                'lr': curr_lr
            })

        if val_r2 >= best_r2 and not config['debug']:
            best_r2 = val_r2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_r2': train_r2,
                'val_loss': val_loss,
                'val_r2': val_r2, 
                'lr': curr_lr
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')

    return train_r2, best_r2


def save_results():
    if os.path.exists('results.csv'):
        with open('results.csv', 'a') as f:
            f.write(f'{run_name}, , {train_r2}, {val_r2}, {test_r2}\n')
    else:
        with open('results.csv', 'w') as f:
            f.write('Run Name, , Train R2, Validation R2, Test R2\n')
            f.write(f'{run_name}, , {train_r2}, {val_r2}, {test_r2}\n')


config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

train_loader, val_loader, test_loader = load_data(config)
model = create_model(config)
criterion, optimizer, scheduler = setup_training(config, model)

if not config['debug']:
    run_name = f'{config["label"]}-{datetime.now().strftime("%m%d_%H%M")}'
    wandb.init(project='IDPBERT', name=run_name)

    save_dir = f'./checkpoints/{run_name}'
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy('./config.yaml', f'{save_dir}/config.yaml')
    shutil.copy('./model/network.py', f'{save_dir}/network.py')

train_r2, val_r2 = train_model()
if not config['debug']:
    model.load_state_dict(torch.load(f'{save_dir}/model.pt')['model_state_dict'], strict=False)
test_r2 = test(model, test_loader, device)
print(f'Test R2 Score: {test_r2:.3f}')
wandb.finish()

if not config['debug']:
    save_results()
