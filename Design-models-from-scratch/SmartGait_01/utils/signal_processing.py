"""
This module contains signal filtering methods required for data preprocessing.
"""

def signal_filtering_gaussian(
    data,
    sigma,
    axis=-1,
    order=0,
    output=None,
    mode='reflect',
    cval=0.0,
    truncate=4.0
):
    from scipy.ndimage import gaussian_filter1d

    signal_filt = gaussian_filter1d(
        input=data.flatten(),
        sigma=sigma,
        axis=axis,
        order=order,
        output=output,
        mode=mode,
        cval=cval,
        truncate=truncate
    )

    return signal_filt