# Copyright 2023, Crown in Right of Canada
'''A skeletal reimplementation of the Google Cloud Storage bucket interface, sufficient to download the files from the public GraphCast GCS bucket.

By default, this module will download files on demand, as needed by graphcast_demo.

If run as a stand-alone script, this module will pre-download all of the files in a specified bucket (default dm_graphcast)
'''

import urllib
import urllib.request
import xml.etree.ElementTree
import dataclasses
import os
import hashlib
import datetime

try:
    # If TQDM is available, use it to show progress bars during
    # file downloads
    import tqdm
    __progress_bar__ = True
except ModuleNotFoundError:
    # Otherwise, don't show progress bars
    __progress_bar__ = False

if (__progress_bar__):
    # From https://github.com/tqdm/tqdm#hooks-and-callbacks
    class TqdmUpTo(tqdm.tqdm):
        def update_to(self,b=1,bsize=1,tsize=None):
            if tsize is not None:
                self.total = tsize
            return (self.update(b*bsize - self.n))

class storage:
    '''Mock storage class'''
    def Client():
        '''Mock function that returns a new MockGCSClient object'''
        return MockGCSClient()

@dataclasses.dataclass(frozen=True)
class BucketEntry:
    '''DataClass that wraps the relevant parts of a file (Contents) entry
    from a GCS bucket'''
    name: str
    hash: str
    size: int
    modified: float # Timestamp


class MockGCSClient():
    '''Class that mocks the .list_blobs and .blob methods of the Google Cloud Storage
    Python module, which is sufficient to download files from a public GCS bucket.'''
    def __init__(self):
        '''Skeletal __init__'''
        self._bucket_name = ''
        self._bucket_url = ''
        self._check_hash = True

    def get_bucket(self,bucket_name,check_hash = True):
        '''Processes the table of contents of the provided public GCS bucket via
        https interface'''
        self._bucket_name = bucket_name
        self._bucket_url = f'https://storage.googleapis.com/{bucket_name}'
        self._check_hash = check_hash

        self._parse_tree()
        return self

    def _parse_tree(self):
        '''Parses the XML representation of a Google Cloud Storage bucket
        contained at url, storing the contents internally'''

        self._bucket_files = []

        with urllib.request.urlopen(self._bucket_url) as bucket_directory:
            xmltree = xml.etree.ElementTree.parse(bucket_directory)
            # Loop over each 'Contents' entry.  Select the entry with '{*}Contents'
            # in order to match regardless of the XML Namespace provided with
            # the bucket
            for bucket_entry in xmltree.findall('{*}Contents'):
                # The filename of the bucket entry is stored in the Key entry; assume
                # that there is only one of them.
                if ((child := bucket_entry.find('{*}Key')) is not None):
                    filename = child.text
                else:
                    # If there is no filename, this is a malformed or unknown entry.
                    # Skip it and continue processing.
                    continue

                # If the filename contains _$folder$, then it's a folder and we can also
                # skip it
                if (filename.find('_$folder$') > 0):
                    continue

                # Last modified time
                if ((child := bucket_entry.find('{*}LastModified')) is not None):
                    # fromisoformat works as of python 3.11, but strptime is necessary with
                    # older versions of Python.  This is brittle if the timestamp format
                    # ever changes
                    last_modified = datetime.datetime.strptime(child.text,'%Y-%m-%dT%H:%M:%S.%f%z').timestamp()
                    #last_modified = datetime.datetime.fromisoformat(child.text).timestamp()
                else:
                    # No last modified time, so assume now?
                    last_modified = datetime.datetime.now().timestamp()

                # File size
                if ((child := bucket_entry.find('{*}Size')) is not None):
                    size = int(child.text)
                else:
                    size = -1

                # File hash
                if ((child := bucket_entry.find('{*}ETag')) is not None):
                    hash = child.text.replace('"','') # Remove double quotes to get the bare hash
                else:
                    hash = ''

                parsed_entry = BucketEntry(name=filename,modified=last_modified,size=size,hash=hash)
                self._bucket_files.append(parsed_entry)
                
    def list_blobs(self,prefix=''):
        '''Return all files in the bucket, optionally matching a provided prefix'''
        return [f for f in self._bucket_files if (f.name.find(prefix)==0)]

    def blob(self,name):
        '''Return a GCSFile_Mock object (with an 'open' method) corresponding to the given blob name'''
        
        url_file = urllib.parse.quote(f'/{name}')
        for b in self._bucket_files:
            if (b.name == name):
                return GCS_MockBlob(b,self._bucket_url + url_file,self._check_hash)
        raise(KeyError(f'{name} not found in GCS bucket'))
        
class GCS_MockBlob():
    '''Class that mocks the 'blob' object returned by the GCS_Mock.blob() method.
    This mock-up class implements functionality sufficient to download the file
    and open it from the local disk.'''
    def __init__(self,blob : BucketEntry, url : str,check_hash : bool):
        self._blob = blob
        self._url = url
        self._check_hash = check_hash

    def _do_download(self):
        global __progress_bar__
        print(f'Downloading from {self._url} ({self._blob.size/1e6:.2f}MB)')

        # If necessary, create the directory that contains this file
        os.makedirs(os.path.dirname(self._blob.name),exist_ok=True)

        if (__progress_bar__):
            with TqdmUpTo(unit='B',unit_scale=True,miniters=1,desc=self._blob.name) as bar:
                urllib.request.urlretrieve(self._url,filename=self._blob.name,reporthook=bar.update_to)
        else:
            urllib.request.urlretrieve(self._url,filename=self._blob.name)

    def _do_verify(self):
        # Check to see whether the blob corresponds to a file on disk

        # Check 1: does the file exist?
        if (not os.path.isfile(self._blob.name)):
            print(f'{self._blob.name} does not exist')
            return False

        # Check 2: does the file have the correct size?
        status = os.stat(self._blob.name)
        if (status.st_size != self._blob.size):
            print(f'{self._blob.name} exists, but it has the wrong size')
            return False

        # Check 3: Has the cloud bucket been modified after
        # the download?
        if (status.st_mtime <= self._blob.modified):
            print(f'{self._blob.name} is older than the last modified time')
            return False

        # Check 3: does the file have the correct hash?
        if (self._check_hash == True):
            md5 = hashlib.md5()
            # The ETag should correspond to an md5sum of the file contents;
            # see https://cloud.google.com/storage/docs/hashes-etags
            with open(self._blob.name,'rb') as f:
                while True:
                    data = f.read(65535)
                    if not data: break
                    md5.update(data)
            if (md5.hexdigest() != self._blob.hash):
                print(f'{self._blob.name} has hash mismatch:\n'+ \
                      f'ours:   {md5.hexdigest()}\n' + \
                      f'theirs: {self._blob.hash}')
                return False
        # All checks pass
        return True
        
    def download(self):
        '''Downloads the file corresponding to this blob, if it does not already exist'''
        if (not self._do_verify()):
            self._do_download()

    def open(self,mode):
        self.download()
        return open(self._blob.name,mode)


# When run as a script, download all of the files in a specified bucket
if (__name__ == '__main__'):
    import sys # Get command-line arguments
    if (len(sys.argv) <= 1):
        bucket_name = 'dm_graphcast'
    else:
        bucket_name = sys.argv[1]

    print(f'Downloading all files in GCS bucket {bucket_name}')

    client = MockGCSClient()
    try:
        client.get_bucket(bucket_name)
    except urllib.error.HTTPError as e:
        print(f'Error downloading {bucket_name}')
        if (e.code == 404):
            print('  Bucket does not exist')
        elif (e.code == 403):
            print('  Bucket is not public')
        else:
            print(f'  {e}')
    for theblob in client._bucket_files:
        client.blob(theblob.name).download()
