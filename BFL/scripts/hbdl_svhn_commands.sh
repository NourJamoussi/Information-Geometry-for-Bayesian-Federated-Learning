###################### Experiments on svhn

echo "Launching experiments on SVHN"

###################### seed 0 

echo "#################### seed = 0 ######################" 

######### FedAvg 

echo "Launching FedAVG using sgd"
python train.py --dataset=svhn --alg=FedAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  

########### BFLAVG 3bl

echo "Launching BFLAVG - HBDL 3bl - using sgd "

echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=3 --update_method='wb_diag'

echo "################## RKL ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=3 --update_method='rkl'

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=3 --update_method='eaa'

echo "################## GAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=3 --update_method='gaa'

echo "################## AALV ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=3 --update_method='aalv'

######### BFLAVG 2bl

echo "Launching BFLAVG - HBDL 2bl - using sgd "

echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=2 --update_method='wb_diag'

echo "################## RKL ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=2 --update_method='rkl'

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=2 --update_method='eaa'

echo "################## GAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=2 --update_method='gaa'

echo "################## AALV ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=2 --update_method='aalv'

########### BFLAVG 1bl

echo "Launching BFLAVG - HBDL 1bl - using sgd "

echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=1 --update_method='wb_diag'

echo "################## RKL ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=1 --update_method='rkl'

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=1 --update_method='eaa'

echo "################## GAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=1 --update_method='gaa'

echo "################## AALV ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=0  --optimizer='sgd'  --nbl=1 --update_method='aalv'


###################### seed 1 

echo "#################### seed = 1 ######################" 

######### FedAvg 

echo "Launching FedAVG using sgd"
python train.py --dataset=svhn --alg=FedAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  

########### BFLAVG 3bl

echo "Launching BFLAVG - HBDL 3bl - using sgd "

echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=3 --update_method='wb_diag'

echo "################## RKL ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=3 --update_method='rkl'

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=3 --update_method='eaa'

echo "################## GAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=3 --update_method='gaa'

echo "################## AALV ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=3 --update_method='aalv'

######### BFLAVG 2bl

echo "Launching BFLAVG - HBDL 2bl - using sgd "

echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=2 --update_method='wb_diag'

echo "################## RKL ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=2 --update_method='rkl'

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=2 --update_method='eaa'

echo "################## GAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=2 --update_method='gaa'

echo "################## AALV ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=2 --update_method='aalv'

########### BFLAVG 1bl

echo "Launching BFLAVG - HBDL 1bl - using sgd "

echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=1 --update_method='wb_diag'

echo "################## RKL ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=1 --update_method='rkl'

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=1 --update_method='eaa'

echo "################## GAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=1 --update_method='gaa'

echo "################## AALV ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=1  --optimizer='sgd'  --nbl=1 --update_method='aalv'



###################### seed 2 

echo "#################### seed = 2 ######################" 

######### FedAvg 

echo "Launching FedAVG using sgd"
python train.py --dataset=svhn --alg=FedAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  

########### BFLAVG 3bl

echo "Launching BFLAVG - HBDL 3bl - using sgd "

echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=3 --update_method='wb_diag'

echo "################## RKL ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=3 --update_method='rkl'

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=3 --update_method='eaa'

echo "################## GAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=3 --update_method='gaa'

echo "################## AALV ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=3 --update_method='aalv'

######### BFLAVG 2bl

echo "Launching BFLAVG - HBDL 2bl - using sgd "

echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=2 --update_method='wb_diag'

echo "################## RKL ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=2 --update_method='rkl'

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=2 --update_method='eaa'

echo "################## GAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=2 --update_method='gaa'

echo "################## AALV ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=2 --update_method='aalv'

########### BFLAVG 1bl

echo "Launching BFLAVG - HBDL 1bl - using sgd "

echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=1 --update_method='wb_diag'

echo "################## RKL ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=1 --update_method='rkl'

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=1 --update_method='eaa'

echo "################## GAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=1 --update_method='gaa'

echo "################## AALV ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs' --init_seed=2  --optimizer='sgd'  --nbl=1 --update_method='aalv'

