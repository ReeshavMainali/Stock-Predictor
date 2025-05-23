# generate_summary_charts.py
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter
import os

def generate_charts(xml_file='test_reports/report.xml', output_dir='test_reports/'):
    if not os.path.exists(xml_file):
        print(f"Error: JUnit XML report '{xml_file}' not found. Generate it first using 'pytest --junitxml={xml_file}'.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tree = ET.parse(xml_file)
    root = tree.getroot() # This is typically <testsuites> or <testsuite>

    results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'error': 0,
        'total': 0
    }
    tests_per_suite = Counter() # For tests per class/file (suite)

    # Handle both single <testsuite> and multiple <testsuites> root elements
    testsuites_element = root if root.tag == 'testsuites' else ET.Element('testsuites')
    if root.tag == 'testsuite':
        testsuites_element.append(root) # Wrap single testsuite for consistent parsing

    for testsuite_node in testsuites_element.findall('testsuite'):
        suite_name = testsuite_node.get('name', 'UnknownSuite')
        # Pytest often includes the full path in the name, try to shorten it
        # e.g., "tests.unit.test_app" or "tests/integration/test_app.py"
        suite_name = suite_name.split('.')[-1] # Get last part after a dot
        suite_name = os.path.basename(suite_name).replace('_', ' ').replace('.py', '') # Get basename and clean up
        
        suite_total = 0
        for testcase_node in testsuite_node.findall('testcase'):
            results['total'] += 1
            suite_total +=1
            if testcase_node.find('failure') is not None:
                results['failed'] += 1
            elif testcase_node.find('skipped') is not None:
                results['skipped'] += 1
            elif testcase_node.find('error') is not None: # JUnit might distinguish errors from failures
                results['error'] += 1
            else:
                results['passed'] += 1
        tests_per_suite[suite_name] += suite_total


    # --- Generate Overall Summary Pie Chart ---
    labels = ['Passed', 'Failed', 'Skipped', 'Error']
    # Filter out categories with 0 tests to avoid cluttering the chart
    active_sizes = [results['passed'], results['failed'], results['skipped'], results['error']]
    active_labels = [label for i, label in enumerate(labels) if active_sizes[i] > 0]
    active_sizes_filtered = [size for size in active_sizes if size > 0]
    
    # Assign colors only to active labels
    color_map = {'Passed': '#4CAF50', 'Failed': '#F44336', 'Skipped': '#FFC107', 'Error': '#9E9E9E'}
    active_colors = [color_map[label] for label in active_labels]
    
    # Explode logic needs to map to the filtered list
    explode_map = {'Passed': 0.05, 'Failed': 0.05, 'Skipped': 0, 'Error': 0}
    active_explode = [explode_map[label] for label in active_labels]


    if results['total'] > 0 and sum(active_sizes_filtered) > 0 :
        plt.figure(figsize=(8, 8))
        plt.pie(active_sizes_filtered, explode=active_explode, labels=active_labels, colors=active_colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title(f"Overall Test Results (Total: {results['total']})")
        plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        pie_chart_path = os.path.join(output_dir, 'test_summary_pie_chart.png')
        plt.savefig(pie_chart_path)
        plt.close()
        print(f"Pie chart saved to {pie_chart_path}")
    else:
        print("No test results with counts > 0 found to generate pie chart.")


    # --- Generate Tests per Suite Bar Chart ---
    if tests_per_suite:
        suite_names = list(tests_per_suite.keys())
        suite_counts = list(tests_per_suite.values())

        plt.figure(figsize=(12, max(6, len(suite_names) * 0.5))) # Adjust height based on number of suites
        bars = plt.barh(suite_names, suite_counts, color='skyblue')
        plt.xlabel('Number of Tests')
        plt.ylabel('Test Suite') # Updated label
        plt.title('Number of Tests Per Suite')
        plt.gca().invert_yaxis() # Display top-to-bottom
        
        # Add text labels for counts on each bar
        for bar in bars:
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                     f'{int(bar.get_width())}', 
                     va='center', ha='left')

        plt.tight_layout() # Adjust layout to prevent labels from being cut off
        bar_chart_path = os.path.join(output_dir, 'tests_per_suite_bar_chart.png')
        plt.savefig(bar_chart_path)
        plt.close()
        print(f"Bar chart saved to {bar_chart_path}")
    else:
        print("No suite data to generate bar chart.")

if __name__ == '__main__':
    # Create the directory if it doesn't exist before calling generate_charts
    # This is good practice, although generate_charts also does it.
    output_directory = 'test_reports'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
        
    generate_charts(xml_file=os.path.join(output_directory, 'report.xml'), output_dir=output_directory)
