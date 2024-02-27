# Change Log
## v1.3.1
> Starting from v1.3, to support auto-detection on Civitai, `calculate_model_hash` will be renamed as 
> `calculate_hash` and it will be enabled by default. Due to the addition of temporary storage of the model hash values,
> the first image generated after switching to the new model will take more time to calculate the hash value, 
> but it will not affect the generation speed afterwards until the server is restarted.

- Add user-friendly warnings and remove the exception raised during the reading of metadata #58 #63
- Add JPEG to the image extension list
- Fix the issue where the `Prompt Saver` node cannot consume filename from another `Prompt Saver` node #62
- Fix the `Prompt Saver` node freezes workflow when name includes `%model` #61
- Remove ckpt validation of the `Prompt Reader` node #59

## v1.3.0
> Starting from this version, to support auto-detection on Civitai, `calculate_model_hash` will be renamed as 
> `calculate_hash` and it will be enabled by default. Due to the addition of temporary storage of the model hash values,
> the first image generated after switching to the new model will take more time to calculate the hash value, 
> but it will not affect the generation speed afterwards until the server is restarted.

- Add `Lora Loader` node and `Lora Selector` node
- Add `VAE_NAME` output to the `Parameter Generator` node and add `vae_name` input to the `Prompt Saver` node #39
- Add `lora_name` input to the `Prompt Saver` node
- Add `resource_hash` switch to the `Prompt Saver` node
- Add resources hashes to metadata for auto-detection on Civitai #35
- Add temporary storage for model hashes
- Add `WEB_DIRECTORY` and remove old js directory
- Fix adding the `Parameter Generator` node to a workflow disables the queue button #45
- Fix the `Parameter Generator` node doesn't load seed value from generated images #44
- Fix the input validation of the `Batch Loader` node #37 #38
- Fix the parsing error caused by the loss of model data #43
- Fix the error message caused by aspect_ratio
- Rename `calculate_model_hash` as `calculate_hash`
- Update seedGen.js to rgthree's latest code
- Update core to cc3c8b2

## v1.2.1
- Fix the input validation of the `Prompt Reader` node #37
- Fix the `Batch Loader` node not working when not connected to any node

## v1.2.0
- Add `Parameter Extractor` node #24
- Add a model matching feature to the `Prompt Reader` node #24
- Add `save_metadata_file` option to the `Prompt Saver` node #30
- Add `FILE_PATH` output to the `Prompt Saver` node #26
- Fix the issue where the `Prompt Merger` node threw an error when merging empty strings #29
- Enhance the `Batch Loader` node to support processing either a single file or a list of files #26
- Update core to 1.3.4.post1

## v1.1.0
- Add `Batch Loader` node #13
- Add `MODEL_NAME` output to the `Prompt Reader` node #23
- Add VAE selector to the `Parameter Generator` node #15
- Add pixel dimensions display to the `aspect_ratio` in the `Parameter Generator` node #6
- Add Positive and Negative Aesthetic Score to the `PARAMETERS` in the `Parameter Generator` node #8
- Add `FILENAME` and `METADATA` output to the `Prompt Saver` node #16

## v1.0.1
- Add a new file-naming mechanism to ensure naming uniqueness
- Fix `%counter` overwriting existing images #11 #14
- Fix the bug causing the `Prompt Saver` node fails to save images in jpg and webp formats #10
- Use relative paths for js imports #12

## v1.0.0
- Initial release