

# This script contains the commands to run the tests on the personalization using models trained with IVON algorithm on different datasets and configurations.


# SVHN 
echo "Running SVHN - WB for personalization"

echo "Running SVHN - WB for update"
python main.py --dataset=svhn --alg=BFLAVG --update_method='wb_diag' --perso_method='wb_diag' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='wb_diag' --perso_method='wb_diag' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='wb_diag' --perso_method='wb_diag' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=2

echo "Running SVHN - RKLB for update"
python main.py --dataset=svhn --alg=BFLAVG --update_method='rkl' --perso_method='wb_diag' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='rkl' --perso_method='wb_diag' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='rkl' --perso_method='wb_diag' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=2

echo "Running SVHN - EAA for update"
python main.py --dataset=svhn --alg=BFLAVG --update_method='eaa' --perso_method='wb_diag' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='eaa' --perso_method='wb_diag' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='eaa' --perso_method='wb_diag' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=2



echo "Running SVHN - RKLB for personalization"

echo "Running SVHN - WB for update"
python main.py --dataset=svhn --alg=BFLAVG --update_method='wb_diag' --perso_method='rkl' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='wb_diag' --perso_method='rkl' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='wb_diag' --perso_method='rkl' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=2

echo "Running SVHN - RKLB for update"
python main.py --dataset=svhn --alg=BFLAVG --update_method='rkl' --perso_method='rkl' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='rkl' --perso_method='rkl' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='rkl' --perso_method='rkl' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=2

echo "Running SVHN - EAA for update"
python main.py --dataset=svhn --alg=BFLAVG --update_method='eaa' --perso_method='rkl' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='eaa' --perso_method='rkl' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=2
python main.py --dataset=svhn --alg=BFLAVG --update_method='eaa' --perso_method='rkl' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=2


# FMNIST 
echo "Running FMNIST - WB for personalization"

echo "Running FMNIST - WB for update"
python main.py --dataset=fmnist --alg=BFLAVG --update_method='wb_diag' --perso_method='wb_diag' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='wb_diag' --perso_method='wb_diag' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='wb_diag' --perso_method='wb_diag' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=5

echo "Running FMNIST - RKLB for update"
python main.py --dataset=fmnist --alg=BFLAVG --update_method='rkl' --perso_method='wb_diag' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='rkl' --perso_method='wb_diag' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='rkl' --perso_method='wb_diag' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=5

echo "Running FMNIST - EAA for update"
python main.py --dataset=fmnist --alg=BFLAVG --update_method='eaa' --perso_method='wb_diag' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='eaa' --perso_method='wb_diag' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='eaa' --perso_method='wb_diag' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=5



echo "Running FMNIST - RKLB for personalization"

echo "Running FMNIST - WB for update"
python main.py --dataset=fmnist --alg=BFLAVG --update_method='wb_diag' --perso_method='rkl' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='wb_diag' --perso_method='rkl' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='wb_diag' --perso_method='rkl' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=5

echo "Running FMNIST - RKLB for update"
python main.py --dataset=fmnist --alg=BFLAVG --update_method='rkl' --perso_method='rkl' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='rkl' --perso_method='rkl' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='rkl' --perso_method='rkl' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=5

echo "Running FMNIST - EAA for update"
python main.py --dataset=fmnist --alg=BFLAVG --update_method='eaa' --perso_method='rkl' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='eaa' --perso_method='rkl' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=5
python main.py --dataset=fmnist --alg=BFLAVG --update_method='eaa' --perso_method='rkl' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=5



# CIFAR10 
echo "Running cifar10 - WB for personalization"

echo "Running cifar10 - WB for update"
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='wb_diag' --perso_method='wb_diag' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='wb_diag' --perso_method='wb_diag' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='wb_diag' --perso_method='wb_diag' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=1

echo "Running cifar10 - RKLB for update"
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='rkl' --perso_method='wb_diag' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='rkl' --perso_method='wb_diag' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='rkl' --perso_method='wb_diag' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=1

echo "Running cifar10 - EAA for update"
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='eaa' --perso_method='wb_diag' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='eaa' --perso_method='wb_diag' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='eaa' --perso_method='wb_diag' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=1



echo "Running cifar10 - RKLB for personalization"

echo "Running cifar10 - WB for update"
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='wb_diag' --perso_method='rkl' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='wb_diag' --perso_method='rkl' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='wb_diag' --perso_method='rkl' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=1

echo "Running cifar10 - RKLB for update"
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='rkl' --perso_method='rkl' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='rkl' --perso_method='rkl' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='rkl' --perso_method='rkl' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=1

echo "Running cifar10 - EAA for update"
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='eaa' --perso_method='rkl' --device='cpu' --init_seed=0  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='eaa' --perso_method='rkl' --device='cpu' --init_seed=1  --optimizer='ivon' --hess_init=1
python main.py --dataset=cifar10 --alg=BFLAVG --update_method='eaa' --perso_method='rkl' --device='cpu' --init_seed=2  --optimizer='ivon' --hess_init=1
