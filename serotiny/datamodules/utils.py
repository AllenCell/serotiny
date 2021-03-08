def subset_channels(channel_subset, channels):
    """
    Subset channels given a list of both
     
    Parameters
    -----------
    channel_subset: List
        List of subset channels

    channels: List
        List of all channels

    Returns:
    channel_indexes: 
        Indexes of subset channels in original channel list
    
    num_channels: 
        New length of channels
    """
    if channel_subset is not None:
        try:
            channel_indexes = [
                channels.index(channel_name) for channel_name in channel_subset
            ]
            num_channels = len(channel_indexes)
        except ValueError:
            raise Exception(
                (
                    f"channel indexes {channel_subset} "
                    f"do not match channel names {channels}"
                )
            )
    return channel_indexes, num_channels
