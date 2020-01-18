import setuptools

from audiodag import MAJOR, MINOR, PATCH


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AudioDAG",  # Replace with your own username
    version=f"{MAJOR}.{MINOR}.{PATCH}",
    author="Gareth Jones",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/garethjns/AudioDAG",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    python_requires='>=3.6',
    install_requires=['numpy', 'seaborn', 'pytest'])

# Lazy auto version increment
PATCH += 1
with open("audiodag/__init__.py", "r") as fh:
    spec = fh.readlines()

new_init = []
for l in spec:
    if l.startswith('PATCH'):
        l = F"PATCH = {PATCH}\n"
    new_init.append(l)

with open("audiodag/__init__.py", "w") as fh:
    fh.writelines(new_init)