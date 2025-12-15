"""Single-command Whisper fine-tuning on local LibriSpeech data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import evaluate
import torch
from datasets import Audio, concatenate_datasets, load_dataset
from transformers import (
    DataCollatorSpeechSeq2SeqWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on LibriSpeech.")
    parser.add_argument("--data-dir", default="data/librispeech/LibriSpeech", help="Path to LibriSpeech root.")
    parser.add_argument("--output-dir", default="outputs/whisper-small-librispeech", help="Where to store checkpoints.")
    parser.add_argument("--model-name", default="openai/whisper-small", help="Base Whisper checkpoint.")
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train.clean.100", "train.clean.360"],
        help="Train splits to concatenate.",
    )
    parser.add_argument("--eval-split", default="validation.clean", help="Eval split.")
    parser.add_argument("--language", default="en", help="Language code for decoder prompt.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if GPU supports.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 if supported.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override total steps (keep -1 for epoch-based).")
    parser.add_argument("--num-proc", type=int, default=min(4, os.cpu_count() or 1), help="Workers for dataset map.")
    return parser.parse_args()


def _prep_datasets(args: argparse.Namespace, processor: WhisperProcessor):
    dataset = load_dataset("librispeech_asr", "all", data_dir=args.data_dir)

    train_parts: List = []
    for split in args.train_splits:
        if split not in dataset:
            raise ValueError(f"Split {split} not found in dataset.")
        train_parts.append(dataset[split])
    train_dataset = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts)
    eval_dataset = dataset[args.eval_split]

    # Ensure consistent audio sampling rate
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    def _prepare(batch):
        audio = batch["audio"]
        features = processor(audio["array"], sampling_rate=audio["sampling_rate"])
        batch["input_features"] = features.input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train_dataset = train_dataset.map(
        _prepare,
        remove_columns=train_dataset.column_names,
        num_proc=args.num_proc,
        desc="Preparing train data",
    )
    eval_dataset = eval_dataset.map(
        _prepare,
        remove_columns=eval_dataset.column_names,
        num_proc=args.num_proc,
        desc="Preparing eval data",
    )
    return train_dataset, eval_dataset


def _build_trainer(args: argparse.Namespace):
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    model.config.suppress_tokens = []
    model.config.use_cache = False  # needed for gradient checkpointing

    train_dataset, eval_dataset = _prep_datasets(args, processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=max(1, args.num_proc - 1),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )
    return trainer


def main() -> None:
    args = _parse_args()
    trainer = _build_trainer(args)
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.push_to_hub = None  # avoid accidental push


if __name__ == "__main__":
    main()
