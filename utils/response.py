from datetime import datetime

class ApiResponse:

    # def __init__(self, success=True, data=None, error=None):
    #     """
    #     Defines the response shape
    #     :param success: A boolean that returns if the request has succeeded or not
    #     :param data: The model's response
    #     :param error: The error in case an exception was raised
    #     """
    #     self.data = data
    #     self.error = error.__str__() if error is not None else ''
    #     self.success = success

    def __init__(self, result=None, replyData=None, error=None):
        """
        Defines the response shape
        :param success: A boolean that returns if the request has succeeded or not
        :param data: The model's response
        :param error: The error in case an exception was raised
        """
        today = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        self.result = result
        self.replyData = error.__str__() if error is not None else replyData
        self.logInfo = error.__str__() if error is not None else 'Process at '+ str(today)