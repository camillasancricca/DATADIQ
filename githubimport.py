from requests import get
from enum import auto, IntEnum
from importlib.machinery import ModuleSpec
from urllib.parse import urljoin
from os.path import join

class GithubImportState(IntEnum):
    USER = auto()
    REPO = auto()
    FILE = auto()

class GithubImportFinder:
    def find_spec(self, modname, path=None, mod=None):
        package, dot, module = modname.rpartition(".")
        if not dot:
            spec = ModuleSpec(
                modname,
                GithubImportLoader(),
                origin="https://github.com/" + module,
                loader_state=GithubImportState.USER,
                is_package=True)
            spec.submodule_search_locations = []
            return spec
        else:
            user, dot2, repo = package.rpartition(".")
            if not dot2:
                spec = ModuleSpec(
                    modname,
                    GithubImportLoader(),
                    origin="https://github.com/" + "/".join([package, module]),
                    loader_state=GithubImportState.REPO,
                    is_package=True)
                spec.submodule_search_locations = []
                return spec
            path = urljoin("https://github.com",
                           join(user, repo, "raw", "master", module + ".py"))
            return ModuleSpec(
                modname,
                GithubImportLoader(),
                origin=path,
                loader_state=GithubImportState.FILE)

class GithubImportLoader:
    def create_module(self, spec):
        return None  # default semantics

    def exec_module(self, module):
        path = module.__spec__.origin
        if module.__spec__.loader_state != GithubImportState.USER:
            setattr(sys.modules[module.__package__],
                    module.__name__.rpartition(".")[-1], module)
        if module.__spec__.loader_state == GithubImportState.FILE:
            # load the module
            code = get(path)
            if code.status_code != 200:
                print(path, code)
                raise ModuleNotFoundError(f"No module named {module.__name__}")
            code = code.text
            exec(code, module.__dict__)

import sys

FINDER = GithubImportFinder()
sys.meta_path.append(FINDER)
