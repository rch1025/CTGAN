"""DataTransformer module."""
from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import OneHotEncoder

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
## ColumnTransformInfo 튜풀 저장: 변수별로 특징을 담고있는 튜플임
# 변환함 (많은 인자들이 추가됨)
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
    'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions', 'original_data'
    ]
)


"""ECDF 적용"""
class DataTransformer(object):

    def __init__(self):
        self.output_info_list = []
        self.output_dimensions = 0
        self._column_transform_info_list = []
        self.dataframe = True

    ### 추가하는 cdf 코드
    def cdf_normalization(self, x, data):  
        return int((data<=x).sum()) / len(data)
    
    #### CDF Normalize가 들어간 연속형 변수 인코더
    def _fit_continuous(self, data):
        column_name = data.columns[0]
        # normalized_data = self.cdf_normalization(data) # self.를 붙여야 함

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=self.cdf_normalization,
            output_info=[SpanInfo(1, 'tanh')], # , SpanInfo(num_components, 'softmax')를 지움
            output_dimensions=1, original_data = data) # original_data 추가


    ## 이산형 변수는 그대로 둠
    def _fit_discrete(self, data):
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories, original_data=None)


    ######## fit()
    def fit(self, raw_data, discrete_columns=()):
        self._column_raw_dtypes = raw_data.infer_objects().dtypes

        # raw_data가 데이터프레임인지 확인하는 것
        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        for idx, column_name in enumerate(raw_data.columns):
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)



    ######## _transform_continuous() : 개별 변수별 반환
    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        cdf_normalizing = column_transform_info.transform # 변환 부분
        transformed = list(map(lambda x: cdf_normalizing(x, data), np.array(data))) # 변환 부분
        # 변환 부분
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed # 변환 부분
        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()



    ######## _synchronous_transform() : 하나씩 변환, 위의 _transform_continous 함수 사용
    def _synchronous_transform(self, raw_data, column_transform_info_list):
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))
        return column_data_list
    
    def _parallel_transform(self, raw_data, column_transform_info_list):
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=20)(processes)
    

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(
                raw_data,
                self._column_transform_info_list
            )
        else:
            column_data_list = self._parallel_transform(
                raw_data,
                self._column_transform_info_list
            )

        return np.concatenate(column_data_list, axis=1).astype(float)


    ### _inverse_transform_continuous()에 들어가는 값
    def cdf_inverse(self, p, data):
        sorted_data = sorted(np.array(data).reshape(-1, 1))
        index = int(p * len(sorted_data))-1
        return sorted_data[index]

    ### 연속형 변수 처리
    def _inverse_transform_continuous(self, column_transform_info, column_data):
        original_data = column_transform_info.original_data
        # data = pd.DataFrame(column_data[:, :1], columns=[column_transform_info.column_name]) # 변환
        
        return list(map(lambda p: self.cdf_inverse(p, original_data), column_data))
    

    ### 이산형 변수 처리
    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]


    ### _inverse_transform_continuous와 _inverse_transform_discrete를 사용해서 변환
    def inverse_transform(self, data):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data)
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)

        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data
    
    """병렬처리 역변환 함수"""
    def _parallel_inverse_transform(self, raw_data):
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = raw_data[:, st:st + dim]
            process = None
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._inverse_transform_continuous)(column_transform_info, column_data)
            else:
                process = delayed(self._inverse_transform_discrete)(column_transform_info, column_data)
            recovered_column_data_list.append(process)
            column_names.append(column_transform_info.column_name)
            st += dim
        recovered_column_data_list = Parallel(n_jobs=20)(recovered_column_data_list) 
        print(len(recovered_column_data_list))
        recovered_data = np.column_stack(recovered_column_data_list)
        print(len(recovered_data))
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()
        return recovered_data
    
    
    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }
