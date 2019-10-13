# Install
    * Ensure pip, setuptools, and wheel are up to date: `python -m pip install --upgrade pip setuptools wheel`

    ## It is highly recommended that you use a python environment to install the libs.

    ## If you want to use your gpu:
        * Download and install CUDA toolkit v10.0 from https://developer.nvidia.com/cuda-toolkit-archive
        * Download and install cuDNN for CUDA toolkit v10.0 from https://developer.nvidia.com/cudnn
        * Run `pip install -r requirements_gpu.txt`

    ## If you want to use your cpu:
        * Just run `pip install -r requirements.txt`

# The Claptcha dataset (additional 256mb download)
    Generated using the claptcha library found on: https://github.com/kuszaj/claptcha
    The code to generate the dataset can be found on: https://github.com/EdYuTo/Captcha-training-AI/tree/master/dataset-generator (note that this code won't generate the dataset on the current folder hierachy)

# The Chars74K dataset (additional 128mb download)
    ## Reference
        The following paper gives further descriptions of this dataset and baseline evaluations using a bag-of-visual-words approach with several feature extraction methods and their combination using multiple kernel learning:

        T. E. de Campos(1), B. R. Babu and M. Varma(2). Character recognition in natural images(3). In Proceedings of the International Conference on Computer Vision Theory and Applications (VISAPP), Lisbon, Portugal, February 2009
        Bibtex(4) | Abstract(5) | PDF(6)

        Follow this link for a list publications that have cited the above paper(7) and this link for papers that mention this dataset(8)

        Files downloaded from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ and reorganized in diffrent folders

        (1) http://personal.ee.surrey.ac.uk/Personal/T.Decampos/
        (2) http://research.microsoft.com/~manik/
        (3) http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf
        (4) https://manikvarma.github.io/pubs/selfbib.html#deCampos09
        (5) https://manikvarma.github.io/pubs/deCampos09-abstract.txt
        (6) http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf
        (7) http://scholar.google.co.uk/scholar?cites=8388314830483915080&as_sdt=2005&sciodt=0,5&hl=en
        (8) http://scholar.google.co.uk/scholar?q=chars74k&hl=en&btnG=Search&as_sdt=1%2C5&as_sdtp=on