class Request:
    ''' One piece of content on given network problem '''

    __slots__ = ('longitude', 'latitude', 'time', 'userid', 'content')
    # Turn off __dict__ to save memory

    def __init__(self, longitude, latitude, time, userid, content):
        ''' Constructor
            @param longitude, latitude : float. Where the request raised
            @param time : datetime. When the request raised
            @param userid : string. Hashed user ID who raised the request
            @param content : int. Content ID (It was once string when initializing)'''

        self.longitude = longitude
        self.latitude = latitude
        self.time = time
        self.userid = userid
        self.content = content
