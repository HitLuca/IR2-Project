import numpy as np


class SyntheticDatasetSDQA:
    @staticmethod
    def generate_batch(batch_size, vocabulary_size):
        points1 = np.zeros((batch_size, vocabulary_size))
        points1[np.random.choice(a=[False, True], size=points1.shape)] = 1
        points1 /= np.linalg.norm(points1, ord=2, axis=1)[:, None]
        points2 = np.zeros((batch_size, vocabulary_size))
        points2[np.random.choice(a=[False, True], size=points2.shape)] = 1
        points2 /= np.linalg.norm(points2, ord=2, axis=1)[:, None]

        labels = np.ones(batch_size) * -1
        for row in range(points1.shape[0]):
            dist = np.linalg.norm(points1[row] - points2[row], ord=2)
            if dist <= 0.999:
                labels[row] = 1
        return points1, points2, labels


class SyntheticDatasetLSTM:
    def generate_batch(self, batch_size, max_length):
        timeseries1 = np.zeros((batch_size, max_length))
        functions1 = []
        for i in range(batch_size):
            timeserie, function = self._sample_function(max_length)
            timeseries1[i, 0:timeserie.shape[0]] = timeserie
            functions1.append(function)

        timeseries2 = np.zeros((batch_size, max_length))
        functions2 = []
        for i in range(batch_size):
            timeserie, function = self._sample_function(max_length)
            timeseries2[i, 0:timeserie.shape[0]] = timeserie
            functions2.append(function)

        y = np.ones(batch_size) * -1
        for i in range(batch_size):
            if functions1[i] == functions2[i]:
                y[i] = 1

        return timeseries1, timeseries2, y

    @staticmethod
    def _sample_function(max_length):
        function_id = np.random.randint(0, 6)
        start = np.random.randint(0, 100)
        end = start + np.random.randint(1, max_length)

        timesteps = np.arange(start, end)
        time_serie = np.empty(timesteps.shape[0])

        for i in range(time_serie.shape[0]):
            if function_id == 0:
                time_serie[i] = np.sin(timesteps[i])
            elif function_id == 1:
                time_serie[i] = np.cos(timesteps[i])
            elif function_id == 2:
                time_serie[i] = np.tan(timesteps[i])
            elif function_id == 3:
                time_serie[i] = np.absolute(timesteps[i])
            elif function_id == 4:
                time_serie[i] = np.square(timesteps[i])
            elif function_id == 5:
                time_serie[i] = np.sqrt(timesteps[i])
        return time_serie, function_id