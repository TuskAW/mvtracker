"""
First download the dataset. You'll have to fill in an online ETH form
and then wait for a few days to get a temporary access code over email.
I used the following sequence of commands to download and unpack the data
into the expected structure. You can probably replace the `dt=...` with
your access token that you can probably find in the access URL (or otherwise
in the page source of the download page that will be linked). Note that
you don't need to download all the data if you don't need it, e.g., maybe
you just want to download a small sample. Note also that in the commands below,
I didn't delete the `*.tar.gz` and `*.zip` files, but you can do so if you'd like.
Note also that the extraction of 00135 had some unexpected structure in that some
takes were in the root of 00135 instead of subfolders, but I ignored that.
```bash
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00122_Inner.tar.gz' -O 00122_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00122_Outer.tar.gz' -O 00122_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00123_Inner.tar.gz' -O 00123_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00123_Outer.tar.gz' -O 00123_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00127_Inner.tar.gz' -O 00127_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00127_Outer.tar.gz' -O 00127_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00129_Inner.tar.gz' -O 00129_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00129_Outer.tar.gz' -O 00129_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00134_Inner.tar.gz' -O 00134_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00134_Outer.tar.gz' -O 00134_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00135_Inner.tar.gz' -O 00135_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00135_Outer_1.tar.gz' -O 00135_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00135_Outer_2.tar.gz' -O 00135_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00136_Inner.tar.gz' -O 00136_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00136_Outer_1.tar.gz' -O 00136_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00136_Outer_2.tar.gz' -O 00136_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00137_Inner_1.tar.gz' -O 00137_Inner_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00137_Inner_2.tar.gz' -O 00137_Inner_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00137_Outer_1.tar.gz' -O 00137_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00137_Outer_2.tar.gz' -O 00137_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00140_Inner_1.tar.gz' -O 00140_Inner_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00140_Inner_2.tar.gz' -O 00140_Inner_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00140_Outer_1.tar.gz' -O 00140_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00140_Outer_2.tar.gz' -O 00140_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00147_Inner.tar.gz' -O 00147_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00147_Outer.tar.gz' -O 00147_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00148_Inner.tar.gz' -O 00148_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00148_Outer.tar.gz' -O 00148_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00149_Inner_1.tar.gz' -O 00149_Inner_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00149_Inner_2.tar.gz' -O 00149_Inner_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00149_Outer_1.tar.gz' -O 00149_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00149_Outer_2.tar.gz' -O 00149_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00151_Inner.tar.gz' -O 00151_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00151_Outer.tar.gz' -O 00151_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00152_Inner.tar.gz' -O 00152_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00152_Outer_1.tar.gz' -O 00152_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00152_Outer_2.tar.gz' -O 00152_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00154_Inner.tar.gz' -O 00154_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00154_Outer_1.tar.gz' -O 00154_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00154_Outer_2.tar.gz' -O 00154_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00156_Inner.tar.gz' -O 00156_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00156_Outer.tar.gz' -O 00156_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00160_Inner.tar.gz' -O 00160_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00160_Outer.tar.gz' -O 00160_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00163_Inner_1.tar.gz' -O 00163_Inner_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00163_Inner_2.tar.gz' -O 00163_Inner_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00163_Outer.tar.gz' -O 00163_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00167_Inner.tar.gz' -O 00167_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00167_Outer.tar.gz' -O 00167_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00168_Inner.tar.gz' -O 00168_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00168_Outer_1.tar.gz' -O 00168_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00168_Outer_2.tar.gz' -O 00168_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00169_Inner.tar.gz' -O 00169_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00169_Outer.tar.gz' -O 00169_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00170_Inner_1.tar.gz' -O 00170_Inner_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00170_Inner_2.tar.gz' -O 00170_Inner_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00170_Outer.tar.gz' -O 00170_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00174_Inner.tar.gz' -O 00174_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00174_Outer.tar.gz' -O 00174_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00175_Inner_1.tar.gz' -O 00175_Inner_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00175_Inner_2.tar.gz' -O 00175_Inner_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00175_Outer_1.tar.gz' -O 00175_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00175_Outer_2.tar.gz' -O 00175_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00176_Inner.tar.gz' -O 00176_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00176_Outer.tar.gz' -O 00176_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00179_Inner.tar.gz' -O 00179_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00179_Outer.tar.gz' -O 00179_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00180_Inner.tar.gz' -O 00180_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00180_Outer.tar.gz' -O 00180_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00185_Inner_1.tar.gz' -O 00185_Inner_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00185_Inner_2.tar.gz' -O 00185_Inner_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00185_Outer_1.tar.gz' -O 00185_Outer_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00185_Outer_2.tar.gz' -O 00185_Outer_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00187_Inner_1.tar.gz' -O 00187_Inner_1.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00187_Inner_2.tar.gz' -O 00187_Inner_2.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00187_Outer.tar.gz' -O 00187_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00188_Inner.tar.gz' -O 00188_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00188_Outer.tar.gz' -O 00188_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00190_Inner.tar.gz' -O 00190_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00190_Outer.tar.gz' -O 00190_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00191_Inner.tar.gz' -O 00191_Inner.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/00191_Outer.tar.gz' -O 00191_Outer.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/Overview.tar.gz' -O Overview.tar.gz
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/README.md' -O README.md
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/4D-DRESS/Template.tar.gz' -O Template.tar.gz

mkdir benchmark
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/Benchmark/Clothing_Recon_inner.zip' -O benchmark/Clothing_Recon_inner.zip
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/Benchmark/Clothing_Recon_outer.zip' -O benchmark/Clothing_Recon_outer.zip
wget 'https://4d-dress.ait.ethz.ch/download.php?dt=def5020078d99c392bec963997126c8af8d41234f84ad3799702aafec5ee264c38b6516a5527a0612a28b607f86221d617d47f2c289c0da697797c694428ca6673011edebc672fe8c769de020df868b99d42d30216ce52086a348d5fc201ec1a421f0bdbaba362d0a19ee346736c6711b492&file=/Benchmark/Human_Recon.zip' -O benchmark/Human_Recon.zip

mkdir -p 00122 00123 00127 00129 00134 00135 00136 00137 00140 00147 00148 00149 00151 00152 00154 00156 00160 00163 00167 00168 00169 00170 00174 00175 00176 00179 00180 00185 00187 00188 00190 00191
tar -xvzf 00122_Inner.tar.gz -C 00122
tar -xvzf 00122_Outer.tar.gz -C 00122

tar -xvzf 00123_Inner.tar.gz -C 00123
tar -xvzf 00123_Outer.tar.gz -C 00123
tar -xvzf 00127_Inner.tar.gz -C 00127
tar -xvzf 00127_Outer.tar.gz -C 00127
tar -xvzf 00129_Inner.tar.gz -C 00129
tar -xvzf 00129_Outer.tar.gz -C 00129
tar -xvzf 00134_Inner.tar.gz -C 00134
tar -xvzf 00134_Outer.tar.gz -C 00134
tar -xvzf 00135_Inner.tar.gz -C 00135
tar -xvzf 00135_Outer_1.tar.gz -C 00135
tar -xvzf 00135_Outer_2.tar.gz -C 00135
tar -xvzf 00136_Inner.tar.gz -C 00136
tar -xvzf 00136_Outer_1.tar.gz -C 00136
tar -xvzf 00136_Outer_2.tar.gz -C 00136
tar -xvzf 00137_Inner_1.tar.gz -C 00137
tar -xvzf 00137_Inner_2.tar.gz -C 00137
tar -xvzf 00137_Outer_1.tar.gz -C 00137
tar -xvzf 00137_Outer_2.tar.gz -C 00137
tar -xvzf 00140_Inner_1.tar.gz -C 00140
tar -xvzf 00140_Inner_2.tar.gz -C 00140
tar -xvzf 00140_Outer_1.tar.gz -C 00140
tar -xvzf 00140_Outer_2.tar.gz -C 00140
tar -xvzf 00147_Inner.tar.gz -C 00147
tar -xvzf 00147_Outer.tar.gz -C 00147
tar -xvzf 00148_Inner.tar.gz -C 00148
tar -xvzf 00148_Outer.tar.gz -C 00148
tar -xvzf 00149_Inner_1.tar.gz -C 00149
tar -xvzf 00149_Inner_2.tar.gz -C 00149
tar -xvzf 00149_Outer_1.tar.gz -C 00149
tar -xvzf 00149_Outer_2.tar.gz -C 00149
tar -xvzf 00151_Inner.tar.gz -C 00151
tar -xvzf 00151_Outer.tar.gz -C 00151
tar -xvzf 00152_Inner.tar.gz -C 00152
tar -xvzf 00152_Outer_1.tar.gz -C 00152
tar -xvzf 00152_Outer_2.tar.gz -C 00152
tar -xvzf 00154_Inner.tar.gz -C 00154
tar -xvzf 00154_Outer_1.tar.gz -C 00154
tar -xvzf 00154_Outer_2.tar.gz -C 00154
tar -xvzf 00156_Inner.tar.gz -C 00156
tar -xvzf 00156_Outer.tar.gz -C 00156
tar -xvzf 00160_Inner.tar.gz -C 00160
tar -xvzf 00160_Outer.tar.gz -C 00160
tar -xvzf 00163_Inner_1.tar.gz -C 00163
tar -xvzf 00163_Inner_2.tar.gz -C 00163
tar -xvzf 00163_Outer.tar.gz -C 00163
tar -xvzf 00167_Inner.tar.gz -C 00167
tar -xvzf 00167_Outer.tar.gz -C 00167
tar -xvzf 00168_Inner.tar.gz -C 00168
tar -xvzf 00168_Outer_1.tar.gz -C 00168
tar -xvzf 00168_Outer_2.tar.gz -C 00168
tar -xvzf 00169_Inner.tar.gz -C 00169
tar -xvzf 00169_Outer.tar.gz -C 00169
tar -xvzf 00170_Inner_1.tar.gz -C 00170
tar -xvzf 00170_Inner_2.tar.gz -C 00170
tar -xvzf 00170_Outer.tar.gz -C 00170
tar -xvzf 00174_Inner.tar.gz -C 00174
tar -xvzf 00174_Outer.tar.gz -C 00174
tar -xvzf 00175_Inner_1.tar.gz -C 00175
tar -xvzf 00175_Inner_2.tar.gz -C 00175
tar -xvzf 00175_Outer_1.tar.gz -C 00175
tar -xvzf 00175_Outer_2.tar.gz -C 00175
tar -xvzf 00176_Inner.tar.gz -C 00176
tar -xvzf 00176_Outer.tar.gz -C 00176
tar -xvzf 00179_Inner.tar.gz -C 00179
tar -xvzf 00179_Outer.tar.gz -C 00179
tar -xvzf 00180_Inner.tar.gz -C 00180
tar -xvzf 00180_Outer.tar.gz -C 00180
tar -xvzf 00185_Inner_1.tar.gz -C 00185
tar -xvzf 00185_Inner_2.tar.gz -C 00185
tar -xvzf 00185_Outer_1.tar.gz -C 00185
tar -xvzf 00185_Outer_2.tar.gz -C 00185
tar -xvzf 00187_Inner_1.tar.gz -C 00187
tar -xvzf 00187_Inner_2.tar.gz -C 00187
tar -xvzf 00187_Outer.tar.gz -C 00187
tar -xvzf 00188_Inner.tar.gz -C 00188
tar -xvzf 00188_Outer.tar.gz -C 00188
tar -xvzf 00190_Inner.tar.gz -C 00190
tar -xvzf 00190_Outer.tar.gz -C 00190
tar -xvzf 00191_Inner.tar.gz -C 00191
tar -xvzf 00191_Outer.tar.gz -C 00191
tar -xvzf Overview.tar.gz
tar -xvzf Template.tar.gz

cd benchmark
unzip Clothing_Recon_inner.zip
unzip Clothing_Recon_outer.zip
unzip Human_Recon.zip
```

With the data downloaded, you can run the script: `python -m scripts.4ddress_preprocessing`.

I create a subselection of the sequences as:
```bash
SRC=datasets/4d-dress-processed-resized-512
DST=datasets/4d-dress-processed-resized-512-selection
mkdir ${DST}

cp ${SRC}/00129_Inner_Take3.pkl ${DST}/00129_Inner_Take3_happy.pkl
cp ${SRC}/00129_Inner_Take4.pkl ${DST}/00129_Inner_Take4_stretch.pkl
cp ${SRC}/00129_Inner_Take5.pkl ${DST}/00129_Inner_Take5_balerina.pkl
cp ${SRC}/00129_Outer_Take13.pkl ${DST}/00129_Outer_Take13_kolo.pkl

cp ${SRC}/00140_Inner_Take8.pkl ${DST}/00140_Inner_Take8_football.pkl
cp ${SRC}/00140_Outer_Take13.pkl ${DST}/00140_Outer_Take13_stretch.pkl
cp ${SRC}/00140_Outer_Take15.pkl ${DST}/00140_Outer_Take15_kicks.pkl

cp ${SRC}/00147_Inner_Take10.pkl ${DST}/00147_Inner_Take10_basketball.pkl
cp ${SRC}/00147_Inner_Take11.pkl ${DST}/00147_Inner_Take11_football.pkl
cp ${SRC}/00147_Outer_Take16.pkl ${DST}/00147_Outer_Take16_dance.pkl
cp ${SRC}/00147_Outer_Take17.pkl ${DST}/00147_Outer_Take17_avatar.pkl

cp ${SRC}/00174_Inner_Take9.pkl ${DST}/00174_Inner_Take9_stretching.pkl

cp ${SRC}/00175_Inner_Take6.pkl ${DST}/00175_Inner_Take6_basketball.pkl
```
"""

import os
import pickle
from typing import Optional

import cv2
import numpy as np
import rerun as rr
import torch
import tqdm
from PIL import Image
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation

from mvtracker.datasets.utils import transform_scene


def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def save_pickle(p, data):
    with open(p, "wb") as f:
        pickle.dump(data, f)


def load_image(path):
    return np.array(Image.open(path))


def extract_4d_dress_data(
        dataset_root: str,
        subject_name: str,
        outfit_name: str,
        take_name,
        save_pkl_path,
        downscaled_longerside: Optional[int] = None,
        save_rerun_viz: bool = True,
        stream_rerun_viz: bool = False,
        skip_if_output_exists: bool = False,
):
    # Skip if output exists
    if skip_if_output_exists and os.path.exists(save_pkl_path):
        print(f"Skipping {save_pkl_path} since it already exists")
        print()
        return save_pkl_path
    else:
        print(f"Processing {save_pkl_path}...")

    base_dir = os.path.join(dataset_root, subject_name, outfit_name, take_name)
    capture_dir = os.path.join(base_dir, "Capture")
    mesh_dir = os.path.join(base_dir, "Meshes_pkl")

    basic_info = load_pickle(os.path.join(base_dir, "basic_info.pkl"))
    scan_frames = basic_info['scan_frames']

    cameras = load_pickle(os.path.join(capture_dir, "cameras.pkl"))
    cam_names = sorted(list(cameras.keys()))

    # Prepare final structure
    rgbs, intrs, extrs, depths = {}, {}, {}, {}
    for cam_name in cam_names:
        rgbs[cam_name] = []
        depths[cam_name] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for frame in tqdm.tqdm(scan_frames, desc="Extracting frame data"):
        mesh_path = os.path.join(mesh_dir, f"mesh-f{frame}.pkl")
        mesh_data = load_pickle(mesh_path)
        vertices = mesh_data["vertices"]
        faces = mesh_data["faces"]

        verts = torch.tensor(vertices, dtype=torch.float32, device=device).unsqueeze(0)
        faces = torch.tensor(faces, dtype=torch.int64, device=device).unsqueeze(0)
        mesh = Meshes(verts=verts, faces=faces)

        for cam_name in cam_names:
            cam_path = os.path.join(capture_dir, cam_name)
            img_path = os.path.join(cam_path, "images", f"capture-f{frame}.png")
            if not os.path.exists(img_path):
                continue

            image = load_image(img_path)
            h, w = image.shape[:2]
            intr = cameras[cam_name]['intrinsics'].copy()
            extr = cameras[cam_name]['extrinsics'].copy()

            if downscaled_longerside is not None:
                scale = downscaled_longerside / max(h, w)
                h, w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                intr[:2] *= scale

            if cam_name not in intrs:
                intrs[cam_name] = intr
                extrs[cam_name] = extr

            rgbs[cam_name].append(image)

            # Convert intrinsics to normalized device coords
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]

            R = extr[:3, :3]
            T = extr[:3, 3]

            R = R.T
            R = R @ np.diag(np.array([-1, -1, 1.]))  # Flip the x and y axes (or multiply f by -1)
            T = T @ np.diag(np.array([-1, -1, 1.]))  # Flip the x and y axes (or multiply f by -1)

            cameras_p3d = PerspectiveCameras(
                focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
                principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
                R=torch.tensor(R, dtype=torch.float32, device=device).unsqueeze(0),
                T=torch.tensor(T, dtype=torch.float32, device=device).unsqueeze(0),
                image_size=torch.tensor([[h, w]], dtype=torch.float32, device=device),
                in_ndc=False,
                device=device,
            )
            raster_settings = RasterizationSettings(
                image_size=(h, w),
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0
            )

            rasterizer = MeshRasterizer(cameras=cameras_p3d, raster_settings=raster_settings)
            fragments = rasterizer(mesh)
            zbuf = fragments.zbuf.squeeze().cpu().numpy()
            zbuf[np.isnan(zbuf)] = 0.0

            depths[cam_name].append(zbuf)

    for cam_name in cam_names:
        if rgbs[cam_name]:
            rgbs[cam_name] = np.stack(rgbs[cam_name]).transpose(0, 3, 1, 2)  # T, C, H, W
            depths[cam_name] = np.stack(depths[cam_name])  # T, H, W

    # Rotate the scene to have the ground at z=0
    rot_x = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    rot_y = Rotation.from_euler('y', 0, degrees=True).as_matrix()
    rot_z = Rotation.from_euler('z', 0, degrees=True).as_matrix()
    rot = torch.from_numpy(rot_z @ rot_y @ rot_x)
    translation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    for cam_name in cam_names:
        extrs[cam_name] = transform_scene(
            1, rot, translation, None, torch.from_numpy(extrs[cam_name][None, None]),
        )[1][0, 0].numpy()

    # Check shapes
    n_frames, _, h, w = rgbs[cam_names[0]].shape
    for cam_name in cam_names:
        assert rgbs[cam_name].shape == (n_frames, 3, h, w)
        assert intrs[cam_name].shape == (3, 3)
        assert extrs[cam_name].shape == (3, 4)

    # Save processed output to a pickle file
    save_pickle(save_pkl_path, dict(
        rgbs=rgbs,
        intrs=intrs,
        extrs=extrs,
        depths=depths,
        ego_cam_name=None,
    ))

    # Visualize the data sample using rerun
    rerun_modes = []
    if stream_rerun_viz:
        rerun_modes += ["stream"]
    if save_rerun_viz:
        rerun_modes += ["save"]
    for rerun_mode in rerun_modes:
        rr.init(f"3dpt", recording_id="v0.16")
        if rerun_mode == "stream":
            rr.connect_tcp()

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.set_time_seconds("frame", 0)
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

        rr.log(
            "mesh",
            rr.Mesh3D(
                vertex_positions=vertices.astype(np.float32),  # (N, 3)
                triangle_indices=faces.cpu().numpy().reshape(-1, 3).astype(np.int32),  # (M, 3)
                albedo_factor=[200, 200, 255],  # Optional color
            ),
        )

        fps = 30
        for frame_idx in range(n_frames):
            rr.set_time_seconds("frame", frame_idx / fps)
            for cam_name in cam_names:
                extr = extrs[cam_name]
                intr = intrs[cam_name]
                img = rgbs[cam_name][frame_idx].transpose(1, 2, 0).astype(np.uint8)
                depth = depths[cam_name][frame_idx]

                h, w = img.shape[:2]
                fx, fy = intr[0, 0], intr[1, 1]
                cx, cy = intr[0, 2], intr[1, 2]

                # Camera pose
                T = np.eye(4)
                T[:3, :] = extr
                world_T_cam = np.linalg.inv(T)
                rr.log(f"{cam_name}/image", rr.Transform3D(
                    translation=world_T_cam[:3, 3],
                    mat3x3=world_T_cam[:3, :3],
                ))
                rr.log(f"{cam_name}/image", rr.Pinhole(
                    image_from_camera=intr,
                    width=w,
                    height=h
                ))
                rr.log(f"{cam_name}/image", rr.Image(img))

                rr.log(f"{cam_name}/depth", rr.Transform3D(
                    translation=world_T_cam[:3, 3],
                    mat3x3=world_T_cam[:3, :3],
                ))
                rr.log(f"{cam_name}/depth", rr.Pinhole(
                    image_from_camera=intr,
                    width=w,
                    height=h
                ))
                rr.log(f"{cam_name}/depth", rr.DepthImage(depth, meter=1.0, colormap="viridis"))

                # Unproject depth to point cloud
                y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                z = depth
                valid = z > 0
                x = x[valid]
                y = y[valid]
                z = z[valid]

                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                pts_cam = np.stack([X, Y, z], axis=-1)

                # Transform to world
                R = world_T_cam[:3, :3]
                t = world_T_cam[:3, 3]
                pts_world = pts_cam @ R.T + t

                # Color
                colors = img[y, x]

                rr.log(f"point_cloud/{cam_name}", rr.Points3D(positions=pts_world, colors=colors))

        if rerun_mode == "save":
            base, name = os.path.split(save_pkl_path)
            name_no_ext = os.path.splitext(name)[0]
            save_rrd_path = os.path.join(base, f"rerun__{name_no_ext}.rrd")
            rr.save(save_rrd_path)
            print(f"Saved rerun viz to {os.path.abspath(save_rrd_path)}")

    print(f"Done with {save_pkl_path}.")
    print()


def crete_overview_pngs(dataset_root, subject_names, overview_dir):
    os.makedirs(overview_dir, exist_ok=True)

    for subject_name in tqdm.tqdm(subject_names):
        if "." in subject_name:
            continue

        for outfit_name in os.listdir(os.path.join(dataset_root, subject_name)):
            if outfit_name not in ["Inner", "Outer"]:
                continue

            for take_name in os.listdir(os.path.join(dataset_root, subject_name, outfit_name)):
                if "." in take_name:
                    continue

                cam_dir = os.path.join(dataset_root, subject_name, outfit_name, take_name, "Capture")
                cam_names = sorted([name for name in os.listdir(cam_dir) if "." not in name])

                first_cam = cam_names[0]
                img_folder = os.path.join(cam_dir, first_cam, "images")
                images = sorted(os.listdir(img_folder))

                last_img = images[-1]
                img_path = os.path.join(img_folder, last_img)

                # Load image and overlay info
                from PIL import Image, ImageDraw, ImageFont
                img = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(img)
                text = (
                    f"{subject_name} / {outfit_name} / {take_name}\n"
                    f"Frame: {last_img.split('-')[-1].split('.')[0]}\n"
                    f"Cams: {cam_names}"
                )

                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
                except:
                    font = ImageFont.load_default()

                draw.text((10, 10), text, fill="white", font=font)

                # Save image
                overview_path = os.path.join(overview_dir, f"{subject_name}__{outfit_name}__{take_name}.png")
                img.save(overview_path)
                print(f"Saved overview to {overview_path}")


def crete_overview_mp4s(dataset_root, subject_names, overview_dir, fps=30):
    os.makedirs(overview_dir, exist_ok=True)

    for subject_name in tqdm.tqdm(subject_names):
        if "." in subject_name:
            continue

        for outfit_name in os.listdir(os.path.join(dataset_root, subject_name)):
            if outfit_name not in ["Inner", "Outer"]:
                continue

            for take_name in os.listdir(os.path.join(dataset_root, subject_name, outfit_name)):
                if "." in take_name:
                    continue

                cam_dir = os.path.join(dataset_root, subject_name, outfit_name, take_name, "Capture")
                cam_names = sorted([name for name in os.listdir(cam_dir) if "." not in name])

                first_cam = cam_names[0]
                img_folder = os.path.join(cam_dir, first_cam, "images")
                images = sorted(os.listdir(img_folder))

                # Load first frame to get size
                first_img = cv2.imread(os.path.join(img_folder, images[0]))
                height, width = first_img.shape[:2]

                video_path = os.path.join(
                    overview_dir,
                    f"{subject_name}__{outfit_name}__{take_name}.mp4"
                )
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                for img_name in images:
                    img_path = os.path.join(img_folder, img_name)
                    img = cv2.imread(img_path)

                    overlay_text = (
                        f"{subject_name} / {outfit_name} / {take_name} | "
                        f"Frame: {img_name.split('-')[-1].split('.')[0]} | "
                        f"Cams: {', '.join(cam_names)}"
                    )
                    cv2.putText(
                        img,
                        overlay_text,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        lineType=cv2.LINE_AA
                    )

                    writer.write(img)

                writer.release()
                print(f"Saved video to {video_path}")


if __name__ == "__main__":
    dataset_root = "datasets/4d-dress"
    output_root = "datasets/4d-dress-processed"
    create_overviews = True  # Creates an overview folder with a png/mp4 summary of each subject-outfit-take

    longside_resolution: Optional[int] = 512
    if longside_resolution is not None:
        output_root += f"-resized-{longside_resolution}"
    os.makedirs(output_root, exist_ok=True)

    subject_names = [
        "00122", "00123", "00127", "00129", "00134",
        "00135", "00136", "00137", "00140", "00147",
        "00148", "00149", "00151", "00152", "00154",
        "00156", "00160", "00163", "00167", "00168",
        "00169", "00170", "00174", "00175", "00176",
        "00179", "00180", "00185", "00187", "00188",
        "00190", "00191",
    ]
    if create_overviews:
        crete_overview_pngs(dataset_root, subject_names, os.path.join(dataset_root, "overview-pngs"))
        crete_overview_mp4s(dataset_root, subject_names, os.path.join(dataset_root, "overview-mp4s"))

    for subject_name in tqdm.tqdm(subject_names):
        if "." in subject_name:
            continue

        for outfit_name in os.listdir(os.path.join(dataset_root, subject_name)):
            if outfit_name not in ["Inner", "Outer"]:
                continue

            for take_name in os.listdir(os.path.join(dataset_root, subject_name, outfit_name)):
                if "." in take_name:
                    continue

                pkl_path = os.path.join(output_root, f"{subject_name}_{outfit_name}_{take_name}.pkl")
                extract_4d_dress_data(
                    dataset_root=dataset_root,
                    subject_name=subject_name,
                    outfit_name=outfit_name,
                    take_name=take_name,
                    downscaled_longerside=longside_resolution,
                    save_pkl_path=pkl_path,
                    save_rerun_viz=True,
                    stream_rerun_viz=False,
                    skip_if_output_exists=True,
                )
