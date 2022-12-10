# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2023-MM-DD
### Added
- import benchmark utils.py
- implement get micro batch, slicing mini batch function 
 
### Changed
- Fixed mini-batching with empty lists as attributes 
### Removed
- Remove internal metrics in favor of `torchmetrics` ([#4287](https://github.com/pyg-team/pytorch_geometric/pull/4287))
