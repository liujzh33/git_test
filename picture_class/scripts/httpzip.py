"""Read members of a remote zip archive through HTTP Range requests.

Only the central directory and the requested members are transferred,
so a single case can be pulled out of a multi-GB archive.
"""
import io
import zipfile

import requests


class HttpFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url = url
        self.session = requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=60)
        self.url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self._cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _get_block(self, idx):
        if idx not in self._cache:
            start = idx * self.block
            end = min(start + self.block, self.size) - 1
            r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[idx] = r.content
        return self._cache[idx]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.block)
            chunk = self._get_block(idx)[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_remote_zip(url):
    return zipfile.ZipFile(HttpFile(url))


if __name__ == "__main__":
    import sys
    z = open_remote_zip(sys.argv[1])
    for info in z.infolist()[: int(sys.argv[2]) if len(sys.argv) > 2 else 60]:
        print(f"{info.file_size/1e6:9.2f} MB  {info.filename}")
    print("total members:", len(z.infolist()))
