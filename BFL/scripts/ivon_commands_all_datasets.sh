
echo "Launching BFL using IVON on FashionMNIST "
echo "################## WB ######################"
python train.py --dataset=fmnist --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=0  --optimizer='ivon' --update_method='wb_diag' --lr=0.1 --n_samples=0 --hess_init=5 --reg=2e-4
python train.py --dataset=fmnist --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=1  --optimizer='ivon' --update_method='wb_diag' --lr=0.1 --n_samples=0 --hess_init=5 --reg=2e-4
python train.py --dataset=fmnist --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=2  --optimizer='ivon' --update_method='wb_diag' --lr=0.1 --n_samples=0 --hess_init=5 --reg=2e-4

echo "################## EAA ######################"
python train.py --dataset=fmnist --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=0  --optimizer='ivon' --update_method='eaa' --lr=0.1 --n_samples=0 --hess_init=5 --reg=2e-4
python train.py --dataset=fmnist --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=1  --optimizer='ivon' --update_method='eaa' --lr=0.1 --n_samples=0 --hess_init=5 --reg=2e-4
python train.py --dataset=fmnist --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=2  --optimizer='ivon' --update_method='eaa' --lr=0.1 --n_samples=0 --hess_init=5 --reg=2e-4

echo "################## RKLB ######################"
python train.py --dataset=fmnist --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=0  --optimizer='ivon' --update_method='rkl' --lr=0.1 --n_samples=0 --hess_init=5 --reg=2e-4
python train.py --dataset=fmnist --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=1  --optimizer='ivon' --update_method='rkl' --lr=0.1 --n_samples=0 --hess_init=5 --reg=2e-4
python train.py --dataset=fmnist --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=2  --optimizer='ivon' --update_method='rkl' --lr=0.1 --n_samples=0 --hess_init=5 --reg=2e-4



echo "Launching BFL using IVON on SVHN "
echo "################## WB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=0  --optimizer='ivon' --update_method='wb_diag' --lr=0.1 --n_samples=0 --hess_init=2 --reg=2e-4
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=1  --optimizer='ivon' --update_method='wb_diag' --lr=0.1 --n_samples=0 --hess_init=2 --reg=2e-4
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=2  --optimizer='ivon' --update_method='wb_diag' --lr=0.1 --n_samples=0 --hess_init=2 --reg=2e-4

echo "################## EAA ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=0  --optimizer='ivon' --update_method='eaa' --lr=0.1 --n_samples=0 --hess_init=2 --reg=2e-4
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=1  --optimizer='ivon' --update_method='eaa' --lr=0.1 --n_samples=0 --hess_init=2 --reg=2e-4
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=2  --optimizer='ivon' --update_method='eaa' --lr=0.1 --n_samples=0 --hess_init=2 --reg=2e-4

echo "################## RKLB ######################"
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=0  --optimizer='ivon' --update_method='rkl' --lr=0.1 --n_samples=0 --hess_init=2 --reg=2e-4
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=1  --optimizer='ivon' --update_method='rkl' --lr=0.1 --n_samples=0 --hess_init=2 --reg=2e-4
python train.py --dataset=svhn --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=2  --optimizer='ivon' --update_method='rkl' --lr=0.1 --n_samples=0 --hess_init=2 --reg=2e-4


echo "Launching BFL using IVON on cifar10 "
echo "################## WB ######################"
python train.py --dataset=cifar10 --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=0  --optimizer='ivon' --update_method='wb_diag' --lr=0.1 --n_samples=0 --hess_init=1 --reg=2e-4
python train.py --dataset=cifar10 --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=1  --optimizer='ivon' --update_method='wb_diag' --lr=0.1 --n_samples=0 --hess_init=1 --reg=2e-4
python train.py --dataset=cifar10 --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=2  --optimizer='ivon' --update_method='wb_diag' --lr=0.1 --n_samples=0 --hess_init=1 --reg=2e-4

echo "################## EAA ######################"
python train.py --dataset=cifar10 --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=0  --optimizer='ivon' --update_method='eaa' --lr=0.1 --n_samples=0 --hess_init=1 --reg=2e-4
python train.py --dataset=cifar10 --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=1  --optimizer='ivon' --update_method='eaa' --lr=0.1 --n_samples=0 --hess_init=1 --reg=2e-4
python train.py --dataset=cifar10 --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=2  --optimizer='ivon' --update_method='eaa' --lr=0.1 --n_samples=0 --hess_init=1 --reg=2e-4

echo "################## RKLB ######################"
python train.py --dataset=cifar10 --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=0  --optimizer='ivon' --update_method='rkl' --lr=0.1 --n_samples=0 --hess_init=1 --reg=2e-4
python train.py --dataset=cifar10 --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=1  --optimizer='ivon' --update_method='rkl' --lr=0.1 --n_samples=0 --hess_init=1 --reg=2e-4
python train.py --dataset=cifar10 --alg=BFLAVG --experiment='noniid-labeldir' --partition='noniid-labeldir' --device='cuda' --process=10 --datadir='./data/' --logdir='./logs/ivon' --init_seed=2  --optimizer='ivon' --update_method='rkl' --lr=0.1 --n_samples=0 --hess_init=1 --reg=2e-4
