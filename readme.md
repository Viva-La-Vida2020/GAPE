# Readme

## Requirements

```
torch
torchvision
numpy
opencv-python
einops
scikit-image
pytorch_lightning
```

## Dataset

The Storm EVent ImagRy (SEVIR) dataset is a collection of temporally and spatially aligned images containing weather events captured by satellite and radar. This dataset was created using publically available datasets distributed by NOAA, including the [GOES-16 geostationary satellite](https://registry.opendata.aws/noaa-goes/), and data derived from [NEXRAD weather radars](https://registry.opendata.aws/noaa-nexrad/), both of which are available on the Registry of Open Data on AWS. This tutorial provides information about the SEVIR dataset as well as code samples that can be used to access the data.

The animation below shows one of thousands of samples in the SEVIR dataset. Each of these "events" consists of 4 hours of data in 5 minute time increments over a 384 km x 384 km patch sampled somewhere over the US. Each event is SEVIR is captured by up to 5 sensors, or image types.

### Downloading SEVIR

If you have [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html), you can download SEVIR using the

```
aws s3 sync --no-sign-request s3://sevir
```

To download only a specific modalitiy, e.g. `vil`, you can instead run

```
aws s3 cp --no-sign-request s3://sevir/CATALOG.csv CATALOG.csv
aws s3 sync --no-sign-request s3://sevir/data/vil .
```

### Data Organization

This section provides descriptions and code samples of how to access, visualize and work with SEVIR data.

SEVIR contains two major components:

- Catalog: A CSV file with rows describing the metadata of an event
- Data Files: A set of HDF5 files containing events for a certain sensor type

We described each component separately below.

### Dataloader

We have wrapped a SEVIR Dataloader with torch-lightning, which provides efficient multi-threaded data loading. Please refer to /sevir_loader/dataloader_example.ipynb

# Run

We recommand to train GAPE with physics-based evolution network freezing, which is easier to converge. Simpy run

```
python train_GAN.py
```

Make sure that the dataloader is correctly initialized and the checkpints loading path is correct.