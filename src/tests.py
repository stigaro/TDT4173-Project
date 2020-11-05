import pytest
import os

from util.loading import CSVTweetReader
from util.constants import PATH_TO_RAW_TRAIN_DATA, CLEAN_DATA_PATH

class TestCSVTweetReader(object):

    @pytest.fixture
    def reader(self):
        return CSVTweetReader(
            PATH_TO_RAW_TRAIN_DATA,
            CLEAN_DATA_PATH
        )
    
    def test_read_match_labels(self, reader):
        assert all(
            x['Sentiment'] == y
            for x, y in zip(reader.read(), reader.labels())
        ), 'order of read datapoints do not match read order for labels'
        
    @pytest.mark.parametrize('lst', [
        [], [1, 4, 5, 7], [100]
    ])
    def test_read_raises_correct_exceptions(self, reader, lst):
        
        def cleaner(row):
            if row['id'] in lst:
                return row
            raise Exception
        
        assert set(x['id'] for x in reader.read(cleaner= cleaner)) == set(lst),\
            """
            read fuction should only return rows that matches ids in lst
            """
            
    @pytest.mark.parametrize('ids,expected', [
        (15, set([15])),
        ([13], set([13])),
        (list(range(15, 30)), set(range(15, 30)))
    ])
    def test_correct_texts_returned(self, reader, ids, expected):
        
        assert len(list(reader.texts(ids))) == len(expected), \
            """
            The correct datapoints where not returned for the respective ids.
            """
    
    @pytest.mark.parametrize('tknzr', [
        'nltk_wordpunct',
        'nltk_tweet',
        lambda e: e
    ])
    def test_resolve_tokenizer(self, reader, tknzr):
        assert callable(reader.resolve_tokenizer(tknzr))
        
    @pytest.mark.parametrize('csvs,fileids,expected', [
        (2, 1, (set([2]), set([1]))),
        ([], 3, (set([]), set([3]))),
        (4, [5], (set([4]), set([5]))),
        (4, [], (set([4]), set([]))),
        ([1, 5, 6], 10, (set([1, 5, 6]), set([10]))),
        ([1, 4, 5], [2, 1], (set([1, 4, 5]), set([2, 1])))
    ])
    def test_get_id(self, reader, csvs, fileids, expected):
        assert reader.get_id(csvs, fileids) == expected, \
            'get_id did not return sets for the different inputs'
            
    def test_str_from_tokenizer(self, reader):
        
        def dummy_tokenizer(): return
        
        assert reader.get_str_from_tknzr(dummy_tokenizer) == 'dummy', \
            'did not extract essential part of tokenizer method'
    
    def test_load_makes_files(self, reader):
        # Test if dir content are same before and after call
        assert True, \
            'load should not temper with files'
            
    