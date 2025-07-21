# Token Tracking Documentation Index

## Documentation Overview

This directory contains comprehensive documentation for Fractalic's enhanced token tracking and analytics system.

## Documents

### 📖 User Documentation

- **[Token Tracking User Guide](token-tracking-user-guide.md)**
  - Getting started with token tracking
  - Understanding token types and costs
  - Tool management and optimization
  - Best practices and troubleshooting

### 🔧 Implementation Documentation

- **[Token Tracking Implementation](token-tracking-implementation.md)**
  - Architecture overview
  - Core components and data models
  - Integration points
  - Configuration and features

### ⚙️ Technical Specifications

- **[Token Tracking Technical Specification](specs/token-tracking-technical-spec.md)**
  - Detailed algorithms and data flow
  - Core implementation patterns
  - Performance optimization
  - Testing strategies

## Quick Start

### For Users
Start with the [User Guide](token-tracking-user-guide.md) to understand how to:
- Monitor token usage in real-time
- Optimize costs through tool selection
- Analyze usage patterns

### For Developers
Review the [Implementation Guide](token-tracking-implementation.md) for:
- System architecture
- Integration patterns
- Configuration options

### For Technical Teams
Check the [Technical Specification](specs/token-tracking-technical-spec.md) for:
- Algorithm details
- Performance considerations
- Testing approaches

## Key Features

### 🎯 Pricing-Aware Analytics
- **Input/Output Separation**: Clear distinction between tokens sent vs received
- **Service Attribution**: MCP vs Fractalic tool cost breakdown
- **Context Analysis**: Message vs schema token identification

### 📊 Real-Time Monitoring
- **Live Token Display**: Real-time usage during operations
- **Cumulative Tracking**: Running totals across session
- **Tool Usage Metrics**: Efficiency and availability tracking

### 💰 Cost Optimization
- **Tool Filtering**: Only pay for tools you use
- **Schema Optimization**: Minimize unnecessary tool definitions
- **Usage Analytics**: Identify optimization opportunities

### 🔍 Session Analytics
- **File-Level Breakdown**: Per-file usage analysis
- **Operation Tracking**: Detailed operation metrics
- **Performance Insights**: Duration and efficiency data

## System Requirements

- **Python 3.8+**
- **LiteLLM**: For accurate token counting
- **Fractalic Core**: Base system components

## Configuration

### Basic Setup
```python
# Enable token tracking
Config.ENABLE_TOKEN_TRACKING = True
```

### Advanced Options
```python
# Configure queue size
Config.TOKEN_QUEUE_SIZE = 1000

# Enable debug output
Config.DEBUG_TOKEN_TRACKING = True

# Set display format
Config.TOKEN_DISPLAY_FORMAT = "enhanced"
```

## Integration Examples

### Simple Usage
```python
from core.token_stats import token_stats

# Your LLM operations automatically tracked
response = llm_client.llm_call(prompt, tools='all')

# View session summary
token_stats.print_session_summary()
```

### Advanced Analytics
```python
# Get detailed breakdown
summary = token_stats.get_session_summary()
file_stats = token_stats.get_stats_by_file()

# Analyze costs
total_input = summary['total_input_tokens']
schema_tokens = summary['total_schema_tokens']
mcp_tokens = summary['total_mcp_schema_tokens']
fractalic_tokens = summary['total_fractalic_schema_tokens']
```

## Support and Troubleshooting

### Common Issues

1. **High Schema Costs**: Review tool selection in [User Guide](token-tracking-user-guide.md#cost-optimization-tips)
2. **Missing Token Data**: Check configuration in [Implementation Guide](token-tracking-implementation.md#configuration)
3. **Performance Issues**: See optimization in [Technical Spec](specs/token-tracking-technical-spec.md#performance-optimization)

### Debug Information

Enable detailed logging:
```python
import logging
logging.getLogger('fractalic.token_tracking').setLevel(logging.DEBUG)
```

### Getting Help

- Review the appropriate documentation above
- Check the troubleshooting sections
- Enable debug logging for detailed information

## Version Information

- **Implementation Version**: 1.0
- **Documentation Version**: 1.0
- **Last Updated**: July 2025

---

*This documentation covers the complete enhanced token tracking system implementation for Fractalic.*
