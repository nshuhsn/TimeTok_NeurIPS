
#UTSD-small_tokenizer without mixed prompt length, with DFA level
exp_num=1000
gpu_id=1
for data_name in ECG5000_pz3; do
    echo "Starting AR Base with data name $data_name"
    nohup python train_var.py data.data_name=$data_name \
            data.tokenizer_name=UTSD_tokenizer \
            data.max_token_size=4096 \
            data.total_vocab_size=4098 \
            data.bos_id=4096 \
            data.mask_id=4097 \
            exp_num=$exp_num \
            gpu_id=$gpu_id \
            validation=teacher_forcing \
            trainer.dataset.batch_size=128 \
            trainer.callbacks.patience=300 \
            trainer.callbacks.max_steps=10000 \
            generate_mix_prompt_length=False \
            ar_model.dim=128 \
            ar_model.depth=2 \
            ar_model.heads=4 \
            ar_model.dim_head=32 \
            ar_model.dropout=0.1 \
            ar_model.generation.inference_temperature=1.0 \
            ar_model.condition_on_class=True \
            ar_model.condition_on_dataset=False  
    sleep 3
done
wait


# Detokenize the generated token sequences
echo "Starting detokenization for ECG5000 using Foundational Tokenizer"
python detokenize_tokens_by_sample.py \
    --dataset_names ECG5000 \
    --tokenizer_name UTSD_tokenizer \
    --exp_num $exp_num \
    --gpu_id $gpu_id \
    --patch_size 3

echo "Done!"
