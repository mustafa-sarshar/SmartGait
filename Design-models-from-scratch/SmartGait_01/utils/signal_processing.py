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
    
    filteredSignal = gaussian_filter1d(
        input=data.flatten(),
        sigma=sigma,
        axis=axis,
        order=order,
        output=output,
        mode=mode,
        cval=cval,
        truncate=truncate
    )
                
    return filteredSignal