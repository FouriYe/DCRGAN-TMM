#apy
python pretrain_SRN.py --gpu 0 --dataset APY --preprocessing --useWD --save_interval 4000 --method all --feat att+vis --rec_attSize 50
python pretrain_SRN.py --gpu 0 --dataset APY --preprocessing --useWD --save_interval 4000 --method all --feat att+vis --rec_attSize 80
#cub
python pretrain_SRN.py --gpu 0 --dataset APY --preprocessing --useWD --save_interval 4000 --method all --feat att+vis --rec_attSize 20
#awa1
python pretrain_SRN.py --gpu 0 --dataset APY --preprocessing --useWD --save_interval 4000 --method all --feat att+vis --rec_attSize 30
python pretrain_SRN.py --gpu 0 --dataset APY --preprocessing --useWD --save_interval 4000 --method all --feat att+vis --rec_attSize 60
#sun
python pretrain_SRN.py --gpu 0 --dataset APY --preprocessing --useWD --save_interval 4000 --method all --feat att+vis --rec_attSize 30
python pretrain_SRN.py --gpu 0 --dataset APY --preprocessing --useWD --save_interval 4000 --method all --feat att+vis --rec_attSize 90