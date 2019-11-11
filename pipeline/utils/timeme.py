import time


def timeme(method):
    def wrapper(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        print(int(round((end_time - start_time) * 1000)), 'ms')
        return result

    return wrapper
