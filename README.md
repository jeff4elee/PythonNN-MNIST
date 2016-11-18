# PythonNN-MNIST
Web app that accepts image uploading. The uploaded images should be a number. The number will be preprocessed (shrinked to 20x20, then bounded by a 28x28 image to comply with MNIST and then normalized) and then read by a neural network that makes use of the MNIST dataset. The preprocessing is currently inaccurate and the data is misinterpreted a third of the time.

<h2><bold> Features </bold></h2>

<ul>
<li> Allows for images to be uploaded </li>
<li> Preprocesses the uploaded image to fit in MNIST standards </li>
<li> Neural net of 728 inputs (1 per pixel) reads the image </li>
<li> Handwriting score (based upon number of evolutions) stored into a sql database </li>
</ul>
