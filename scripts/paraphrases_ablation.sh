#!/bin/bash
# Script per ablation study su numero di parafrasi e epoche
# GPU: Ada Generation RTX 5000 (16GB)
# Modello: Llama-3.2-1B-Instruct (più leggero)
# Split: forget10 / retain90

# Crea cartella logs se non esiste
mkdir -p logs

# Redirect tutto su file log con data
LOG_FILE="logs/ablation_$(date +%Y%m%d).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Disabilita verifica SSL (necessario per Zscaler proxy)
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export PYTHONHTTPSVERIFY=0

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
echo "Log salvato in: $LOG_FILE"
echo "SSL verification: DISABLED (Zscaler proxy compatibility)"

# Parametro per forzare sovrascrittura (default: false)
FORCE_OVERWRITE=false

# Parse argomenti CLI
for arg in "$@"; do
    case $arg in
        --force=true|--force)
            FORCE_OVERWRITE=true
            shift
            ;;
    esac
done

# Configurazione fissa
model="Llama-3.2-1B-Instruct"
forget_split="forget10"
retain_split="retain90"
holdout_split="holdout10"
num_epochs=20

# File parafrasi
paraphrases_file="data/tofu_${forget_split}_augmented.json"
paraphrases_generator="scripts/generation/generate_paraphrases.py"

# Array per ordine garantito di esecuzione
trainers_order=("GradDiff" "NPO" "SimNPO" "DPO" "RMU" "UNDIAL")

# Metodi di unlearning da testare
declare -A trainers_experiments=(
    ["GradDiff"]="unlearn/tofu/paraphrased.yaml"
    ["NPO"]="unlearn/tofu/paraphrased.yaml"
    ["SimNPO"]="unlearn/tofu/paraphrased.yaml"
    ["DPO"]="unlearn/tofu/idk_paraphrased.yaml"
    ["RMU"]="unlearn/tofu/paraphrased.yaml"
    ["UNDIAL"]="unlearn/tofu/paraphrased.yaml"
)

# Numero di parafrasi da testare (0 = solo domanda originale)
paraphrase_counts=(0 5 10 15 20)

# Batch size ottimizzato per 16GB GPU
per_device_train_batch_size=4
gradient_accumulation_steps=8

# Path del modello base
model_path="open-unlearning/tofu_${model}_full"
retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

echo "=========================================="
echo "Paraphrases Ablation Study"
echo "=========================================="
echo "Modello: ${model}"
echo "Split: ${forget_split} / ${retain_split}"
echo "Epoche: ${num_epochs}"
echo "Parafrasi da testare: ${paraphrase_counts[@]} (0=baseline senza parafrasi)"
echo "Metodi: ${trainers_order[@]}"
echo "Force overwrite: ${FORCE_OVERWRITE}"
echo "=========================================="
# Controlla se esiste il file delle parafrasi
if [ ! -f "${paraphrases_file}" ]; then
    echo ""
    echo "⚠️  File parafrasi non trovato: ${paraphrases_file}"
    echo "📝 Generazione automatica delle parafrasi in corso..."
    echo ""
    
    # Genera 20 parafrasi per ogni domanda (massimo richiesto dallo studio)
    CUDA_VISIBLE_DEVICES=0 python ${paraphrases_generator} \
        --dataset locuslab/TOFU \
        --split ${forget_split} \
        --output ${paraphrases_file} \
        --num-paraphrases 20 \
        --model meta-llama/Llama-3.2-3B-Instruct
    
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Generazione parafrasi fallita!"
        echo "   Genera manualmente con:"
        echo "   python ${paraphrases_generator} --split ${forget_split} --output ${paraphrases_file} --num-paraphrases 20"
        exit 1
    fi
    
    echo ""
    echo "✅ Parafrasi generate con successo!"
    echo "   File: ${paraphrases_file}"
    echo ""
else
    echo ""
    echo "✅ File parafrasi trovato: ${paraphrases_file}"
    
    # Verifica quante parafrasi contiene
    num_paras=$(jq '.[0].paraphrases | length' ${paraphrases_file} 2>/dev/null || echo "unknown")
    echo "   Parafrasi per domanda: ${num_paras}"
    
    if [ "${num_paras}" != "unknown" ] && [ "${num_paras}" -lt 20 ]; then
        echo ""
        echo "⚠️  Attenzione: Il file contiene solo ${num_paras} parafrasi per domanda."
        echo "   Lo studio richiede 20 parafrasi per la configurazione para20."
        echo "   Considera di rigenerare il file con:"
        echo "   python ${paraphrases_generator} --split ${forget_split} --output ${paraphrases_file} --num-paraphrases 20"
        echo ""
    fi
fi
# Loop su numero di parafrasi
for num_paraphrases in "${paraphrase_counts[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing with ${num_paraphrases} paraphrases"
    echo "=========================================="
    
    # Loop sui metodi di unlearning (ordine garantito)
    for trainer in "${trainers_order[@]}"; do
        experiment="${trainers_experiments[$trainer]}"
        
        task_name="tofu_${model}_${forget_split}_${trainer}_para${num_paraphrases}_ep${num_epochs}"
        save_dir="saves/unlearn/${task_name}"
        
        # Verifica se il training esiste già
        if [ -d "${save_dir}" ] && [ "${FORCE_OVERWRITE}" != "true" ]; then
            echo ""
            echo "⏭️  SKIP: ${task_name} already exists"
            echo "    Use './paraphrases_ablation.sh --force' to overwrite"
            continue
        fi
        
        # Se forziamo overwrite, rimuovi directory esistente
        if [ -d "${save_dir}" ] && [ "${FORCE_OVERWRITE}" = "true" ]; then
            echo ""
            echo "🗑️  OVERWRITE: Removing existing ${task_name}"
            rm -rf "${save_dir}"
        fi
        
        echo ""
        echo ">>> Training: ${task_name}"
        echo "    Trainer: ${trainer}"
        echo "    Experiment: ${experiment}"
        echo "    Paraphrases: ${num_paraphrases}"
        echo "    Epochs: ${num_epochs}"
        
        # UNLEARNING - Gestione speciale per DPO
        if [ "${trainer}" = "DPO" ]; then
            # DPO usa TOFU_QA_forget_augmented_idk
            CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
                experiment=${experiment} \
                trainer=${trainer} \
                task_name=${task_name} \
                model=${model} \
                forget_split=${forget_split} \
                retain_split=${retain_split} \
                model.model_args.pretrained_model_name_or_path=${model_path} \
                retain_logs_path=${retain_logs_path} \
                data.forget.TOFU_QA_forget_augmented_idk.args.num_paraphrases_per_question=${num_paraphrases} \
                data.forget.TOFU_QA_forget_augmented_idk.args.include_original=true \
                trainer.args.num_train_epochs=${num_epochs} \
                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                trainer.args.per_device_eval_batch_size=2 \
                trainer.args.eval_on_start=false \
                trainer.args.eval_strategy=epoch \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true
        else
            # Altri metodi usano TOFU_QA_forget_augmented
            CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
                experiment=${experiment} \
                trainer=${trainer} \
                task_name=${task_name} \
                model=${model} \
                forget_split=${forget_split} \
                retain_split=${retain_split} \
                model.model_args.pretrained_model_name_or_path=${model_path} \
                retain_logs_path=${retain_logs_path} \
                data.forget.TOFU_QA_forget_augmented.args.num_paraphrases_per_question=${num_paraphrases} \
                data.forget.TOFU_QA_forget_augmented.args.include_original=true \
                trainer.args.num_train_epochs=${num_epochs} \
                trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
                trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
                trainer.args.per_device_eval_batch_size=2 \
                trainer.args.eval_on_start=false \
                trainer.args.eval_strategy=epoch \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true
        fi
        
        if [ $? -ne 0 ]; then
            echo "❌ ERROR: Training failed for ${task_name}"
            continue
        fi
        
        echo "✅ Training completed: ${task_name}"
    done
done

echo ""
echo "=========================================="
echo "Ablation Study Completed!"
echo "=========================================="
echo ""
echo "Risultati salvati in: saves/unlearn/"
echo ""
echo "Per generare i grafici, usa:"
echo "python scripts/plot_ablation_results.py"
