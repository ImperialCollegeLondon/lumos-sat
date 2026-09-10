# Guowang / HULIANWANG DIGUI Brightness Model

This example contains the final surface-based optical brightness model developed for an MSc individual project on **Space Object Brightness Modelling** at Imperial College London.

The model predicts the apparent optical brightness of the first ten HULIANWANG DIGUI (Guowang) satellites using the Lumos-Sat framework and compares the predictions with ground-based SCORE observations.

## Final Model

The spacecraft is represented using:

- A simplified trapezoidal-prism spacecraft body
- Six reflecting body surfaces
- Two Sun-tracking solar arrays
- Front and rear reflecting surfaces for each solar array
- A Binomial BRDF for the spacecraft body
- A Lambertian BRDF for the solar arrays
- Direct sunlight and Earthshine
- Conversion of predicted reflected intensity to AB magnitude

The same model configuration is applied to DIGUI-01 through DIGUI-10 without satellite-specific retuning.

## Main Script

The final model is contained in:
examples/guowang_final_brightness_model.py

