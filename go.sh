#!/bin/bash

PWD=`pwd`
export PYTHONPATH=$PYTHONPATH:$PWD

#==============================================
#python /data/qcao/tools/subword-nmt/subword_nmt/learn_joint_bpe_and_vocab.py --input exper/corpus.tc.de exper/corpus.tc.en -s 32000 -o bpe32k --write-vocabulary vocab.de vocab.en
#python /data/qcao/tools/subword-nmt/subword_nmt/apply_bpe.py --vocabulary vocab.de --vocabulary-threshold 50 -c bpe32k < exper/corpus.tc.de > exper/corpus.tc.32k.de
#python /data/qcao/tools/subword-nmt/subword_nmt/apply_bpe.py --vocabulary vocab.en --vocabulary-threshold 50 -c bpe32k < exper/corpus.tc.en > exper/corpus.tc.32k.en
#python /data/qcao/tools/subword-nmt/subword_nmt/apply_bpe.py --vocabulary vocab.de --vocabulary-threshold 50 -c bpe32k < exper/newstest2014.tc.de > exper/newstest2014.tc.32k.de
#python /data/qcao/tools/subword-nmt/subword_nmt/apply_bpe.py --vocabulary vocab.en --vocabulary-threshold 50 -c bpe32k < exper/newstest2014.tc.en > exper/newstest2014.tc.32k.en
#python /data/qcao/tools/subword-nmt/subword_nmt/apply_bpe.py --vocabulary vocab.de --vocabulary-threshold 50 -c bpe32k < exper/newstest2015.tc.de > exper/newstest2015.tc.32k.de
#python thumt/scripts/shuffle_corpus.py --corpus exper/corpus.tc.32k.de exper/corpus.tc.32k.en --suffix shuf
#python thumt/scripts/build_vocab.py exper/corpus.tc.32k.de.shuf vocab.32k.de
#python thumt/scripts/build_vocab.py exper/corpus.tc.32k.en.shuf vocab.32k.en
#CUDA_VISIBLE_DEVICES=2 nohup python thumt/bin/trainer.py --input exper/corpus.tc.32k.de.shuf exper/corpus.tc.32k.en.shuf --vocabulary vocab.32k.de.txt vocab.32k.en.txt --model transformer --validation exper/newstest2014.tc.32k.de --references exper/newstest2014.tc.32k.en --parameters=batch_size=4096,device_list=[0],train_steps=200000 > log.train &
#==============================================

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python thumt/bin/trainer.py --input experiment/train.zh.bpe experiment/train.en.bpe --vocabulary experiment/vocab.zh.txt experiment/vocab.en.txt --model transformer --validation experiment/nist06.cn.bpe --references experiment/nist06.en0.bpe experiment/nist06.en1.bpe experiment/nist06.en2.bpe experiment/nist06.en3.bpe --parameters=batch_size=4096,device_list=[0,1,2,3],train_steps=200000 > log.train &

#CUDA_VISIBLE_DEVICES=0 nohup python thumt/bin/translator.py --models transformer --input experiment/nist06.cn.bpe --output experiment/valid.trans --vocabulary experiment/vocab.zh.txt experiment/vocab.en.txt --checkpoints train/eval --parameters=device_list=[0],top_beams=1 > log.test &

#sed -r 's/(@@ )|(@@ ?$)//g' < experiment/valid.trans > experiment/valid.trans.norm
