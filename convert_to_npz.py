import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def slices(slices_in,   # the 2D slices
           titles=None, # list of titles
           suptitle = None,
           cmaps=None,  # list of colormaps
           norms=None,  # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,  # option to plot the images in a grid or a single row
           width=15,    # width in in
           show=True,   # option to actually show the plot (plt.show())
           axes_off=True,
           save=False,  # option to save plot
           save_path=None,  # save path
           imshow_args=None):
    '''
    plot a grid of slices (2d images)
    '''
    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, 'each slice has to be 2d or RGB (3 channels)'
        slices_in[si] = slice_in.astype('float')

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    if show:
        plt.tight_layout()
        plt.show()

    if save:
        plt.savefig(save_path + '.png')
    return (fig, axs)

def rescale(itkimg):
    """
    Normalize image intensity to [0, 1].
    """
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(itkimg)
    return sitk.IntensityWindowing(itkimg,
                                    windowMinimum=minmax.GetMinimum(),
                                    windowMaximum=minmax.GetMaximum(),
                                    outputMinimum=0.0,
                                    outputMaximum=1.0)

def crop_ndarray(ndarr_img, uppoint=[0, 13, 13], out_size=[160, 160, 192], show=False):
    """
    Crop a 3D numpy image to specified output size from given upper-left coordinate.
    """
    cropped = ndarr_img[
        uppoint[0]:uppoint[0] + out_size[0],
        uppoint[1]:uppoint[1] + out_size[1],
        uppoint[2]:uppoint[2] + out_size[2]
    ]

    if show:
        mid_slices = [np.take(cropped, cropped.shape[d] // 2, axis=d) for d in range(3)]
        mid_slices[1] = np.rot90(mid_slices[1], 1)
        mid_slices[2] = np.rot90(mid_slices[2], -1)
        slices(mid_slices, cmaps=['gray'], grid=[1, 3], show=True)

    return cropped

def construct_npz(save_path, img_path, seg_path, synth_seg_path, age, disease_condition,
                  uppoint=[7, 37, 48], out_size=[208, 176, 160], show=False):
    """
    Constructs a .npz file from NIfTI image, segmentation, and synthseg, along with metadata.

    Parameters:
    - save_path: Path where the .npz file will be saved.
    - img_path: Path to the input image (.nii.gz), e.g., align_norm.nii.gz.
    - seg_path: Path to the segmentation file (.nii.gz), e.g., align_aseg.nii.gz.
    - synth_seg_path: Path to the synthseg segmentation (.nii.gz).
    - age: Float representing the subject's age.
    - disease_condition: Integer representing diagnosis (e.g., 0 for HC, 1 for AD).
    - uppoint: Starting coordinates for cropping (the default uppoint assumes input sizes are 256x256x256).
    - out_size: Output size for cropped image.
    - show: Whether to show cropped slices for visual inspection.
    """
    if os.path.exists(save_path):
        print(f"{save_path} already exists. Skipping.")
        return

    # Load and normalize input image
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img = rescale(img)
    np_img = sitk.GetArrayFromImage(img)
    print(np_img.shape)

    # Load segmentations
    seg = sitk.ReadImage(seg_path, sitk.sitkInt16)
    np_seg = sitk.GetArrayFromImage(seg)
    print(np_seg.shape)

    synth_seg = sitk.ReadImage(synth_seg_path, sitk.sitkInt16)
    np_synth = sitk.GetArrayFromImage(synth_seg)
    print(np_synth.shape)

    # Crop all arrays
    np_img = crop_ndarray(np_img, uppoint=uppoint, out_size=out_size, show=show)
    np_seg = crop_ndarray(np_seg, uppoint=uppoint, out_size=out_size, show=show)
    np_synth = crop_ndarray(np_synth, uppoint=uppoint, out_size=out_size, show=show)
    print(np_img.shape)
    print(np_seg.shape)
    print(np_synth.shape)

    # Save to compressed npz
    np.savez_compressed(
        save_path,
        vol=np_img,
        seg=np_seg,
        synth_seg=np_synth,
        age=age,
        disease_condition=disease_condition
    )
    print(f"Saved: {save_path}")

# -------------------- MAIN --------------------

if __name__ == "__main__":
    # Path to the input normalized image (.nii.gz), e.g., align_norm.nii.gz
    img_path = "./example/OAS30001_d0129.nii.gz"

    # Path to the anatomical segmentation (.nii.gz), e.g., align_aseg.nii.gz
    seg_path = "./example/OAS30001_d0129_seg.nii.gz"

    # Path to the synthseg result (.nii.gz), usually ran on align_norm.nii.gz
    synth_seg_path = "./example/OAS30001_d0129_SynthSeg.nii.gz"

    # Output path to save the .npz file
    save_path = "./example/OAS30001_d0129_saved.npz"

    # Metadata: Age of the subject. => CHANGE THIS TO REAL AGE! <=
    age = 65.54792466

    # Metadata: Disease condition (0 for HC, 1 for AD, etc.). => CHANGE THIS TO REAL CONDITION! <=
    disease_condition = 0

    # Run preprocessing and save npz
    construct_npz(save_path, img_path, seg_path, synth_seg_path, age, disease_condition, show=True)