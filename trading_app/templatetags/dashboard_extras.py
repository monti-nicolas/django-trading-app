from django import template
from django.utils.safestring import mark_safe
import json

register = template.Library()


@register.filter(name='zip')
def zip_lists(a, b):
    """
    Zip two lists together for iteration in templates.

    Usage in template:
    {% for item1, item2 in list1|zip:list2 %}
        {{ item1 }} - {{ item2 }}
    {% endfor %}
    """
    return zip(a, b)


@register.filter(name='zip3')
def zip_three_lists(a, b_c):
    """
    Zip three lists together for iteration in templates.
    This is a workaround since Django template filters can only take 2 arguments.

    Usage in template:
    First zip two lists, then zip the result with the third:
    {% for headline, label, score in headlines|zip:labels|zip:scores %}
        {{ headline }} - {{ label }} - {{ score }}
    {% endfor %}

    Or use the custom three-way zip:
    Pass the second and third list as a tuple.
    """
    if isinstance(b_c, tuple) and len(b_c) == 2:
        b, c = b_c
        return zip(a, b, c)
    return zip(a, b_c)


@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Get an item from a dictionary using a variable key.

    Usage in template:
    {{ mydict|get_item:key_variable }}
    """
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None


@register.filter(name='multiply')
def multiply(value, arg):
    """
    Multiply a value by an argument.

    Usage in template:
    {{ price|multiply:quantity }}
    """
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0


@register.filter(name='divide')
def divide(value, arg):
    """
    Divide a value by an argument.

    Usage in template:
    {{ total|divide:count }}
    """
    try:
        if float(arg) == 0:
            return 0
        return float(value) / float(arg)
    except (ValueError, TypeError):
        return 0


@register.filter(name='percentage')
def percentage(value, total):
    """
    Calculate percentage of value relative to total.

    Usage in template:
    {{ value|percentage:total }}
    """
    try:
        if float(total) == 0:
            return 0
        return (float(value) / float(total)) * 100
    except (ValueError, TypeError):
        return 0


@register.filter(name='abs_value')
def abs_value(value):
    """
    Return the absolute value.

    Usage in template:
    {{ number|abs_value }}
    """
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return 0


@register.filter(name='format_large_number')
def format_large_number(value):
    """
    Format large numbers with K, M, B suffixes.

    Usage in template:
    {{ large_number|format_large_number }}

    Examples:
    1234 -> 1.2K
    1234567 -> 1.2M
    1234567890 -> 1.2B
    """
    try:
        value = float(value)
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.1f}K"
        else:
            return f"{value:.0f}"
    except (ValueError, TypeError):
        return value


@register.filter(name='to_json')
def to_json(value):
    """
    Convert a Python object to JSON string.
    Useful for passing data to JavaScript.

    Usage in template:
    <script>
        var data = {{ mydata|to_json|safe }};
    </script>
    """
    try:
        return mark_safe(json.dumps(value))
    except (TypeError, ValueError):
        return '{}'


@register.filter(name='dict_to_json')
def dict_to_json(value):
    """
    Convert a dictionary to a pretty-printed JSON string.

    Usage in template:
    {{ mydict|dict_to_json }}
    """
    try:
        return json.dumps(value, indent=2)
    except (TypeError, ValueError):
        return '{}'


@register.filter(name='get_sentiment_color')
def get_sentiment_color(label):
    """
    Get Bootstrap color class based on sentiment label.

    Usage in template:
    <span class="badge bg-{{ label|get_sentiment_color }}">{{ label }}</span>
    """
    sentiment_colors = {
        'positive': 'success',
        'negative': 'danger',
        'neutral': 'secondary',
        'bullish': 'success',
        'bearish': 'danger',
    }
    return sentiment_colors.get(label.lower(), 'secondary')


@register.filter(name='get_trend_icon')
def get_trend_icon(value):
    """
    Get Bootstrap icon based on trend direction.

    Usage in template:
    <i class="bi bi-{{ value|get_trend_icon }}"></i>
    """
    try:
        value = float(value)
        if value > 0:
            return 'arrow-up-circle-fill text-success'
        elif value < 0:
            return 'arrow-down-circle-fill text-danger'
        else:
            return 'dash-circle text-secondary'
    except (ValueError, TypeError):
        return 'dash-circle text-secondary'


@register.filter(name='is_positive')
def is_positive(value):
    """
    Check if a value is positive.

    Usage in template:
    {% if value|is_positive %}
        <span class="text-success">Gain</span>
    {% else %}
        <span class="text-danger">Loss</span>
    {% endif %}
    """
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False


@register.filter(name='is_negative')
def is_negative(value):
    """
    Check if a value is negative.

    Usage in template:
    {% if value|is_negative %}
        <span class="text-danger">Loss</span>
    {% endif %}
    """
    try:
        return float(value) < 0
    except (ValueError, TypeError):
        return False


@register.simple_tag
def get_kpi_trend_class(current, previous):
    """
    Get CSS class for KPI trend indication.

    Usage in template:
    {% get_kpi_trend_class current_value previous_value as trend_class %}
    <div class="{{ trend_class }}">...</div>
    """
    try:
        current = float(current)
        previous = float(previous)
        if current > previous:
            return 'trend-up text-success'
        elif current < previous:
            return 'trend-down text-danger'
        else:
            return 'trend-neutral text-secondary'
    except (ValueError, TypeError):
        return 'trend-neutral text-secondary'


@register.simple_tag
def calculate_change_percent(current, previous):
    """
    Calculate percentage change between two values.

    Usage in template:
    {% calculate_change_percent current_price previous_price as change_pct %}
    {{ change_pct }}%
    """
    try:
        current = float(current)
        previous = float(previous)
        if previous == 0:
            return 0
        change = ((current - previous) / previous) * 100
        return round(change, 2)
    except (ValueError, TypeError):
        return 0


@register.inclusion_tag('trading_app/components/sentiment_badge.html')
def sentiment_badge(label, score):
    """
    Render a sentiment badge component.

    Usage in template:
    {% sentiment_badge sentiment_label sentiment_score %}
    """
    return {
        'label': label,
        'score': score,
        'color_class': get_sentiment_color(label)
    }


@register.inclusion_tag('trading_app/components/kpi_card_mini.html')
def kpi_card_mini(title, value, icon=None, color='primary'):
    """
    Render a mini KPI card component.

    Usage in template:
    {% kpi_card_mini "Total Value" total_value "graph-up" "success" %}
    """
    return {
        'title': title,
        'value': value,
        'icon': icon,
        'color': color
    }


@register.filter(name='default_if_none_or_nan')
def default_if_none_or_nan(value, default='N/A'):
    """
    Return default value if value is None, NaN, or invalid.

    Usage in template:
    {{ value|default_if_none_or_nan:"Not Available" }}
    """
    try:
        if value is None:
            return default
        float_value = float(value)
        # Check for NaN
        if float_value != float_value:  # NaN != NaN is True
            return default
        return value
    except (ValueError, TypeError):
        return default


@register.filter(name='safe_floatformat')
def safe_floatformat(value, decimals=2):
    """
    Safely format a float, returning a default if conversion fails.

    Usage in template:
    {{ value|safe_floatformat:2 }}
    """
    try:
        float_value = float(value)
        # Check for NaN
        if float_value != float_value:
            return 'N/A'
        return f"{float_value:.{decimals}f}"
    except (ValueError, TypeError):
        return 'N/A'


@register.filter(name='list_length')
def list_length(value):
    """
    Get the length of a list or queryset.

    Usage in template:
    {{ mylist|list_length }}
    """
    try:
        return len(value)
    except (TypeError, AttributeError):
        return 0


@register.simple_tag
def define(value):
    """
    Define a variable in template.

    Usage in template:
    {% define some_calculation as my_var %}
    {{ my_var }}
    """
    return value


@register.filter(name='index')
def index(sequence, position):
    """
    Get item at index from a sequence.

    Usage in template:
    {{ mylist|index:0 }}
    """
    try:
        return sequence[int(position)]
    except (IndexError, KeyError, ValueError, TypeError):
        return None