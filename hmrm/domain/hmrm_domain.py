import numpy as np
import pandas as pd
from numpy.linalg import norm
from numpy.linalg import inv as inverse
import scipy.sparse as sparse


class HmrmDomain:
    def __init__(self):
        self._user_location_frequency = np.array([])
        self._user_time_frequency = np.array([])
        self._location_co_ocurrency = np.array([])
        self._location_time = np.array([])
        self._weight = 0.001
        self.activity_location = np.array([])
        self.activity_time = np.array([])
        self.user_activity = np.array([])
        self.activity_embedding = np.array([])
        self.target_location_embedding = np.array([])
        self.context_location_embedding = np.array([])
        self.time_slot_embedding = np.array([])

    def _create_user_location_frequency_matrix(self, users_checkins):
        placeids = users_checkins["placeid"].tolist()
        userids = users_checkins["userid"]
        total_users = len(users_checkins["userid"].unique())
        total_places = len(users_checkins["placeid"].unique())
        print(total_places, total_users)

        self._user_location_frequency = sparse.lil_matrix((total_users, total_places))

        for i in range(len(placeids)):
            self._user_location_frequency[userids[i], placeids[i]] += 1

    def _create_user_time_frequency_matrix(self, users_checkins: pd.DataFrame):
        users_checkins_sorted = users_checkins.sort_values(by=["datetime"])

        users_ids = users_checkins_sorted["userid"]
        datetimes = pd.to_datetime(users_checkins["datetime"])
        total_users = len(users_checkins["userid"].unique())

        self._user_time_frequency = np.zeros((total_users, 48))

        for i, j in zip(users_ids, datetimes):
            if j.weekday() >= 5:
                self._user_time_frequency[i][j.hour + 24] += 1
            else:
                self._user_time_frequency[i][j.hour] += 1

    def _create_location_coocurrency_matrix(self, users_checkins):
        try:
            users_checkins_sorted = users_checkins.sort_values(by=["datetime"])
            locations = users_checkins_sorted["placeid"].tolist()
            number_of_locations = len(users_checkins["placeid"].unique())

            self._location_co_ocurrency = sparse.lil_matrix(
                (number_of_locations, number_of_locations)
            )  ##location co occurency represents memory for save memory

            for i in range(len(locations)):
                for j in range(1, 6):
                    if (i - j) < 0:
                        break
                    self._location_co_ocurrency[locations[i], locations[i - j]] += 1
                for j in range(1, 6):
                    if (i + j) > len(locations) - 1:
                        break
                    self._location_co_ocurrency[locations[i], locations[j + i]] += 1
            sum_of_dl = np.sum(self._location_co_ocurrency)
            l_occurrency = np.sum(self._location_co_ocurrency, axis=1).reshape(-1, 1)
            c_occurrency = np.sum(self._location_co_ocurrency, axis=0).reshape(1, -1)

            for i in range(number_of_locations):
                line = self._location_co_ocurrency[i].toarray()
                ##PMI em subdivisoes da matriz esparsa
                self._location_co_ocurrency[i] = np.maximum(
                    np.log2(
                        np.maximum(line * sum_of_dl, 1)
                        / (l_occurrency[i] * c_occurrency)
                    ),
                    0,
                )

        except Exception as e:
            raise e

    def _create_location_time_matrix(self, users_checinks):
        locations = users_checinks["placeid"].tolist()
        datetimes = users_checinks["datetime"].tolist()
        total_locations = len(users_checinks["placeid"].unique())
        Dt = np.zeros((total_locations, 48))

        for i in range(len(locations)):
            if datetimes[i].weekday() >= 5:
                Dt[locations[i]][datetimes[i].hour + 24] += 1
            else:
                Dt[locations[i]][datetimes[i].hour] += 1

        sum_of_dt = np.sum(Dt)
        l_occurrency = np.sum(Dt, axis=1).reshape(-1, 1)
        c_occurrency = np.sum(Dt, axis=0).reshape(1, -1)

        mult = l_occurrency * c_occurrency
        mult[mult == 0] = -1

        tmp = np.maximum(Dt * sum_of_dt, 1) / mult
        tmp[tmp < 0] = 0
        self._location_time = np.maximum(np.log2(tmp), 0)

    def _objective_function(self, l2_weight):
        def first_component(l2_weight):
            first_equation = l2_weight * norm(
                (
                    self._user_location_frequency
                    - np.dot(self.user_activity, self.activity_location.T)
                )
            )

            second_equation = (1 - l2_weight) * norm(
                (
                    self._user_time_frequency
                    - np.dot(self.user_activity, self.activity_time.T)
                )
            )
            return first_equation + second_equation

        def second_component(l2_weight):
            first_equation = l2_weight * norm(
                (
                    self._location_co_ocurrency
                    - np.dot(
                        self.target_location_embedding,
                        self.context_location_embedding.T,
                    )
                )
            )
            second_equation = (1 - l2_weight) * norm(
                (
                    self._location_time
                    - np.dot(self.target_location_embedding, self.time_slot_embedding.T)
                )
            )
            return first_equation + second_equation

        def third_component(l2_weight):
            first_equation = l2_weight * norm(
                (
                    self.activity_location
                    - np.dot(self.context_location_embedding, self.activity_embedding.T)
                )
            )
            second_equation = (1 - l2_weight) * norm(
                (
                    self.activity_time
                    - np.dot(self.time_slot_embedding, self.activity_embedding.T)
                )
            )
            return first_equation + second_equation

        activity_modeling_component = first_component(l2_weight)
        trajectory_embedding_component = second_component(l2_weight)
        collaborative_learning_component = third_component(l2_weight)

        objective_function = (
            activity_modeling_component
            + trajectory_embedding_component
            + collaborative_learning_component
        )
        objective_function += self._weight * norm(self.user_activity)
        objective_function += self._weight * norm(self.activity_time)
        objective_function += self._weight * norm(self.activity_embedding)
        objective_function += self._weight * norm(self.activity_location)
        objective_function += self._weight * norm(self.context_location_embedding)
        objective_function += self._weight * norm(self.target_location_embedding)
        objective_function += self._weight * norm(self.time_slot_embedding)

        return objective_function

    def _initialize_parameters(self, checkins, K, M):
        total_locations = len(checkins["placeid"].unique())
        total_users = len(checkins["userid"].unique())
        time_slot = 48

        self.activity_location = np.random.normal(size=(total_locations, K))
        self.activity_time = np.random.normal(size=(time_slot, K))
        self.user_activity = np.random.normal(size=(total_users, K))
        self.activity_embedding = np.random.normal(size=(K, M))
        self.target_location_embedding = np.random.normal(size=(total_locations, M))
        self.context_location_embedding = np.random.normal(size=(total_locations, M))
        self.time_slot_embedding = np.random.normal(size=(time_slot, M))

    def user_activity_embedding_function(self, K, l2_weight):
        first_equation = (
            l2_weight * (self._user_location_frequency * self.activity_location)
        ) + ((1 - l2_weight) * np.dot(self._user_time_frequency, self.activity_time))
        second_equation = (
            l2_weight * np.dot(self.activity_location.T, self.activity_location)
        ) + (
            (1 - l2_weight) * np.dot(self.activity_time.T, self.activity_time)
            + (l2_weight * np.identity(K))
        )
        return np.dot(first_equation, inverse(second_equation))

    def acticity_location_embedding_function(self, K, l2_weight):
        first_equation = l2_weight * (
            (self._user_location_frequency.T * self.user_activity)
            + np.dot(self.context_location_embedding, self.activity_embedding.T)
        )
        second_equation = (
            l2_weight * np.dot(self.user_activity.T, self.user_activity)
        ) + ((self._weight + l2_weight) * np.identity(K))
        return np.dot(first_equation, inverse(second_equation))

    def activity_time_embedding_function(self, K, l2_weight):
        first_equation = (1 - l2_weight) * (
            np.dot(self._user_time_frequency.T, self.user_activity)
            + np.dot(self.time_slot_embedding, self.activity_embedding.T)
        )
        second_equation = (1 - l2_weight) * (
            np.dot(self.user_activity.T, self.user_activity)
            + (1 - self._weight + l2_weight) * np.identity(K)
        )
        return np.dot(first_equation, inverse(second_equation))

    def activity_embedding_function(self, M, l2_weight):
        first_equation = (
            l2_weight
            * np.dot(self.activity_location.T, self.context_location_embedding)
        ) + ((1 - l2_weight) * np.dot(self.activity_time.T, self.time_slot_embedding))
        second_equation = (
            (
                l2_weight
                * np.dot(
                    self.context_location_embedding.T, self.context_location_embedding
                )
            )
            + (
                (1 - l2_weight)
                * np.dot(self.time_slot_embedding.T, self.time_slot_embedding)
            )
            + (self._weight * np.identity(M))
        )
        return np.dot(first_equation, inverse(second_equation))

    def target_location_embedding_function(self, M, l2_weight):
        first_equation = (
            l2_weight * self._location_co_ocurrency * self.context_location_embedding
        ) + ((1 - l2_weight) * np.dot(self._location_time, self.time_slot_embedding))

        second_equation = (
            (
                l2_weight
                * np.dot(
                    self.context_location_embedding.T, self.context_location_embedding
                )
            )
            + (
                (1 - l2_weight)
                * np.dot(self.time_slot_embedding.T, self.time_slot_embedding)
            )
            + (self._weight * np.identity(M))
        )

        return np.dot(first_equation, inverse(second_equation))

    def context_location_embedding_function(self, M, l2_weight):
        first_equation = l2_weight * (
            self._location_co_ocurrency.T * self.target_location_embedding
            + np.dot(self.activity_location, self.activity_embedding)
        )
        second_equation = (
            l2_weight
            * (
                np.dot(self.target_location_embedding.T, self.target_location_embedding)
                + np.dot(self.activity_embedding.T, self.activity_embedding)
            )
        ) + (self._weight * np.identity(M))
        return np.dot(first_equation, inverse(second_equation))

    def time_slot_embedding_function(self, M, l2_weight):
        first_equation = (1 - l2_weight) * (
            np.dot(self._location_time.T, self.target_location_embedding)
            + np.dot(self.activity_time, self.activity_embedding)
        )
        second_equation = (
            (1 - l2_weight)
            * (
                np.dot(self.target_location_embedding.T, self.target_location_embedding)
                + np.dot(self.activity_embedding.T, self.activity_embedding)
            )
        ) + (self._weight * np.identity(M))
        return np.dot(first_equation, inverse(second_equation))

    def _optimize_parameters(self, K, M, l2_weight):
        self.user_activity = self.user_activity_embedding_function(K, l2_weight)
        self.user_activity[self.user_activity < 0] = 0

        self.activity_location = self.acticity_location_embedding_function(K, l2_weight)
        self.activity_location[self.activity_location < 0] = 0

        self.activity_time = self.activity_time_embedding_function(K, l2_weight)
        self.activity_time[self.activity_time < 0] = 0

        self.activity_embedding = self.activity_embedding_function(M, l2_weight)
        self.target_location_embedding = self.target_location_embedding_function(
            M, l2_weight
        )
        self.context_location_embedding = self.context_location_embedding_function(
            M, l2_weight
        )
        self.time_slot_embedding = self.time_slot_embedding_function(M, l2_weight)

    def start(self, checkins, l2_weight=0.1, K=10, M=100):
        checkins["datetime"] = pd.to_datetime(checkins["datetime"])

        self._create_user_location_frequency_matrix(checkins)
        self._create_location_coocurrency_matrix(checkins)
        self._create_user_time_frequency_matrix(checkins)
        self._create_location_time_matrix(checkins)

        self._initialize_parameters(checkins, K, M)

        value = 100000

        for i in range(1):
            print(i)
            self._optimize_parameters(K, M, l2_weight)
            objective_func = self._objective_function(l2_weight)

            # print("user activity:", self.user_activity) # theta
            # print("activity location:", self.activity_location) # Al
            # print("activity time:", self.activity_time) # At
            # print("activity embedding:", self.activity_embedding) # Ea
            # print("target location embedding:", self.target_location_embedding) # El
            # print("context location embedding:", self.context_location_embedding) # Ec
            # print("time slot embedding:", self.time_slot_embedding) # Et

            if (value - objective_func) <= 0.1:
                break
            value = objective_func
