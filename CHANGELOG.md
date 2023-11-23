# Change Log
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