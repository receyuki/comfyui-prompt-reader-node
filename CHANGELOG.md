# Change Log
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