
from extended_watermark_processor import WatermarkLogitsProcessor

def test(tokenizer, model, input_text):
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                gamma=0.25,
                                                delta=2.0,
                                                seeding_scheme="selfhash") #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
    # Note:
    # You can turn off self-hashing by setting the seeding scheme to `minhash`.

    tokenized_input = tokenizer(input_text).to(model.device)
    # note that if the model is on cuda, then the input is on cuda
    # and thus the watermarking rng is cuda-based.
    # This is a different generator than the cpu-based rng in pytorch!

    output_tokens = model.generate(**tokenized_input,
                                logits_processor=LogitsProcessorList([watermark_processor]))

    # if decoder only model, then we need to isolate the
    # newly generated tokens as only those are watermarked, the input/prompt is not
    output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]