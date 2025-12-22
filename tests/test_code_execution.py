from matharena.tools.code_execution import CodeRunner, execute_code, DockerCodeRunner


def test_basic():
    """Basic test that prints 'Hello, World!'."""
    code_runner = CodeRunner()
    docker_code_runner = DockerCodeRunner()
    result = code_runner.execute_python_code("print('Hello, World!')", exec_timeout=60)
    result_docker = docker_code_runner.execute_python_code("print('Hello, World!')", exec_timeout=60)
    code_runner.terminate()
    assert result["stdout"] == "Hello, World!\n"
    assert result["stderr"] == ""
    assert result_docker["stdout"] == "Hello, World!\n"
    assert result_docker["stderr"] == ""


def test_cpp_basic():
    """Basic test that prints 'Hello, World!'"""
    code_runner = CodeRunner()
    docker_code_runner = DockerCodeRunner()
    result = code_runner.execute_cpp_code(
        """#include<iostream>\nint main() { std::cout << "Hello, World!" << std::endl; return 0; }""", exec_timeout=60
    )
    result_docker = docker_code_runner.execute_cpp_code(
        """#include<iostream>\nint main() { std::cout << "Hello, World!" << std::endl; return 0; }""", exec_timeout=60
    )
    code_runner.terminate()
    assert result["stdout"] == "Hello, World!\n"
    assert result["stderr"] == ""
    assert result_docker["stdout"] == "Hello, World!\n"
    assert result_docker["stderr"] == ""


def test_libraries():
    """Test that the libraries are installed correctly."""
    code_runner = CodeRunner()
    docker_code_runner = DockerCodeRunner()
    result = code_runner.execute_python_code("import numpy as np; print(np.sum(np.array([1, 2, 3])))", exec_timeout=60)
    result_docker = docker_code_runner.execute_python_code(
        "import numpy as np; print(np.sum(np.array([1, 2, 3])))", exec_timeout=60
    )

    code_runner.terminate()
    assert result["stdout"] == "6\n"
    assert result["stderr"] == ""
    assert result_docker["stdout"] == "6\n"
    assert result_docker["stderr"] == ""


def test_time():
    """Test that the time is measured correctly."""
    code_runner = CodeRunner()
    result = code_runner.execute_python_code("import time; time.sleep(10); print('Hello, World!')", exec_timeout=20)
    result_docker = DockerCodeRunner().execute_python_code(
        "import time; time.sleep(10); print('Hello, World!')", exec_timeout=20
    )
    code_runner.terminate()
    assert result["stdout"] == "Hello, World!\n"
    assert result["stderr"] == ""
    assert result["time"] >= 9 and result["time"] <= 11
    assert result_docker["stdout"] == "Hello, World!\n"
    assert result_docker["stderr"] == ""
    assert result_docker["time"] >= 9 and result_docker["time"] <= 11

def test_timeout():
    """Test that the timeout is enforced."""
    code_runner = CodeRunner()
    result = code_runner.execute_python_code("import time; time.sleep(10); print('This should timeout')", exec_timeout=2)
    result_docker = DockerCodeRunner().execute_python_code(
        "import time; time.sleep(10); print('This should timeout')", exec_timeout=2
    )
    code_runner.terminate()
    assert "exceeded the timeout" in result["stderr"]
    assert "exceeded the timeout" in result_docker["stderr"]


def test_globalfn_timeout_wrapper():

    def short_code_execution(code, lang):
        return execute_code(code, lang, exec_timeout=2)

    code = "import time; time.sleep(10); print('This should timeout')"
    lang = "python"
    out = short_code_execution(code, lang)
    assert "exceeded the timeout of 2 seconds" in out
