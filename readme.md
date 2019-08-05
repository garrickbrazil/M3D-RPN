# M3D-RPN: Monocular 3D Region Proposal Network for Object Detection

Garrick Brazil, Xiaoming Liu

## Introduction


Monocular 3D region proposal network source code as detailed in [arXiv report](https://arxiv.org/abs/1907.06038), accepted to ICCV 2019 (Oral). Please also visit our [project page](http://cvlab.cse.msu.edu/project-m3d-rpn.html).

Our framework is implemented and tested with Ubuntu 16.04, CUDA 8.0, Python 3, NVIDIA 1080 Ti GPU. Unless otherwise stated the below scripts and instructions assume working directory is the project root. 

If you utilize this framework, please cite our ICCV 2019 paper. 

    @inproceedings{brazil2019m3drpn,
        title={M3D-RPN: Monocular 3D Region Proposal Network for Object Detection},
        author={Brazil, Garrick and Liu, Xiaoming},
        booktitle={Proceedings of the IEEE International Conference on Computer Vision},
        address={Seoul, South Korea},
        year={2019}
    }
    

## Setup

- **Cuda & Python**

    In this project we utilize Pytorch with Python 3, Cuda 8, and a few Anaconda packages. Please review and follow this [installation guide](setup.md). However, feel free to try alternative versions or modes of installation. 

- **Data**

    Download the full [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) detection dataset. Then place a softlink (or the actual data) in  *M3D-RPN/data/kitti*. 

	```
    cd M3D-RPN
	ln -s /path/to/kitti data/kitti
	```

	Then use the following scripts to extract the data splits, which use softlinks to the above directory for efficient storage. 

    ```
    python data/kitti_split1/setup_split.py
    python data/kitti_split2/setup_split.py
    ```
    
    Next, build the KITTI devkit eval for each split.

	```
	sh data/kitti_split1/devkit/cpp/build.sh
	sh data/kitti_split2/devkit/cpp/build.sh
	```
    
    Lastly, build the nms modules
    
    ```
	cd lib/nms
	make
	```

## Training

We use [visdom](https://github.com/facebookresearch/visdom) for visualization and graphs. Optionally, start the server by command line

```
python -m visdom.server -port 8100 -readonly
```
The port can be customized in *scripts/config* files. The training monitor can be viewed at [http://localhost:8100](http://localhost:8100)

Training is split into a warmup and main configurations. Review the configurations in *scripts/config* for details. 

``` 
// First train the warmup (without depth-aware)
python scripts/train_rpn_3d.py --config=kitti_3d_multi_warmup

// Then train the main experiment (with depth-aware)
python scripts/train_rpn_3d.py --config=kitti_3d_multi_main
```

If your training is accidentally stopped, you can resume at a checkpoint based on the snapshot with the *restore* flag. 
For example to resume training starting at iteration 10k, use the following command.

```
python scripts/train_rpn_3d.py --config=kitti_3d_multi_main --restore=10000
```

## Testing

We provide models for the main experiments on val1 / val2 / test data splits available to download here [M3D-RPN-Release.zip](https://www.cse.msu.edu/computervision/M3D-RPN-Release.zip).

Testing requires paths to the configuration file and model weights, exposed variables near the top *scripts/test_rpn_3d.py*. To test a configuration and model, simply update the variables and run the test file as below. 

```
python scripts/test_rpn_3d.py 
```

## Contact
For questions regarding M3D-RPN, feel free to post here or directly contact the authors {[brazilga](http://garrickbrazil.com), [liuxm](http://www.cse.msu.edu/~liuxm/index2.html)}@msu.edu.
