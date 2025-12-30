Plans:

A PipeLine that:

    1. Split an Image (e.g. 2x2, 3x3, ...)
    2. Process each Image by gemini
        - Denoise, sharpen, ...
    3. Threshold the Image
    4. Analyze the image:
        - feed the full thresholed image to gemini to color it
        - Fit circles, eclipses on splitted image
        - output a csv
    5. Stitch the images back together

This pipeline is for the image analysis.

There might be other ones as well, right?

Let's assume the current v3 is working fine!

**_ We should try it on different types of images! _**

## It is so good if we show the output from a variaty of images!

# This way we say it is general and not limited to one thing!

The next steps now:

    1. Build the final pipeline,
    2. Test and run it,
    3. Create a Config file

    4. Generate some outputs based on different images.
    5. Wrap the whole thing on a review and test loop?
    6. I can try image auto adjust as the first step.

The first 3 can be completed tomorrow.
