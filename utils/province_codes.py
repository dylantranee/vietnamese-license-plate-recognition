"""
Vietnamese Province Code Mapping

This module provides a mapping between Vietnamese license plate province codes
and their corresponding province/city names. Province codes are 2-digit numbers
(11-99) that appear at the beginning of every Vietnamese license plate.

Note: Some provinces have multiple codes due to historical changes or high
vehicle registration volumes (e.g., Hanoi has codes 29-33, 40).

Usage:
    from utils.province_codes import get_province_name
    
    province = get_province_name("51")  # Returns "TP. Hồ Chí Minh"
"""

# Province code to name mapping
# Format: "code": "Province/City Name"
PROVINCE_CODES = {
    # Northern Vietnam
    "11": "Cao Bằng",
    "12": "Lạng Sơn",
    "14": "Quảng Ninh",
    "99": "Bắc Ninh", "98": "Bắc Ninh",
    "22": "Tuyên Quang", "23": "Tuyên Quang",
    "24": "Lào Cai", "21": "Lào Cai",
    "25": "Lai Châu",
    "27": "Điện Biên",
    "26": "Sơn La",
    "20": "Thái Nguyên", "97": "Thái Nguyên",
    "19": "Phú Thọ", "28": "Phú Thọ", "88": "Phú Thọ",
    
    # Hanoi (multiple codes due to high registration volume)
    "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội", "32": "Hà Nội", "33": "Hà Nội", "40": "Hà Nội",
    
    # Hai Phong and surrounding areas
    "15": "Hải Phòng", "16": "Hải Phòng", "34": "Hải Phòng",
    "89": "Hưng Yên", "17": "Hưng Yên",
    "35": "Ninh Bình", "18": "Ninh Bình", "90": "Ninh Bình",
    
    # Central Vietnam
    "36": "Thanh Hoá",
    "37": "Nghệ An",
    "38": "Hà Tĩnh",
    "74": "Quảng Trị", "73": "Quảng Trị",
    "75": "Thừa Thiên Huế",
    "43": "Đà Nẵng", "92": "Đà Nẵng",
    "76": "Quảng Ngãi", "82": "Quảng Ngãi",
    
    # Central Highlands
    "81": "Gia Lai", "77": "Gia Lai",
    "47": "Đắk Lắk", "78": "Đắk Lắk",
    "79": "Khánh Hoà", "85": "Khánh Hoà",
    "49": "Lâm Đồng", "48": "Lâm Đồng", "86": "Lâm Đồng",
    
    # Southern Vietnam
    "60": "Đồng Nai", "39": "Đồng Nai", "93": "Đồng Nai",
    
    # Ho Chi Minh City (multiple codes due to very high registration volume)
    "41": "TP. Hồ Chí Minh", "50": "TP. Hồ Chí Minh", "51": "TP. Hồ Chí Minh", "52": "TP. Hồ Chí Minh", 
    "53": "TP. Hồ Chí Minh", "54": "TP. Hồ Chí Minh", "55": "TP. Hồ Chí Minh", "56": "TP. Hồ Chí Minh", 
    "57": "TP. Hồ Chí Minh", "58": "TP. Hồ Chí Minh", "59": "TP. Hồ Chí Minh", "61": "TP. Hồ Chí Minh", "72": "TP. Hồ Chí Minh",
    
    # Mekong Delta
    "70": "Tây Ninh", "62": "Tây Ninh",
    "66": "Đồng Tháp", "63": "Đồng Tháp",
    "65": "Cần Thơ", "83": "Cần Thơ", "95": "Cần Thơ",
    "64": "Vĩnh Long", "71": "Vĩnh Long", "84": "Vĩnh Long",
    "69": "Cà Mau", "94": "Cà Mau",
    "68": "An Giang", "67": "An Giang",
    
    # Special code for traffic police
    "80": "Cục Cảnh sát giao thông"
}

def get_province_name(code):
    """
    Returns the province name for a given code.
    
    Args:
        code: Province code as string or int (e.g., "51" or 51)
        
    Returns:
        Province/city name if found, "Unknown" otherwise
        
    Examples:
        >>> get_province_name("51")
        'TP. Hồ Chí Minh'
        >>> get_province_name(30)
        'Hà Nội'
        >>> get_province_name("99")
        'Bắc Ninh'
    """
    return PROVINCE_CODES.get(str(code), "Unknown")
