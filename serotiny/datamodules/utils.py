def subset_channels(channel_subset, channels):
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
